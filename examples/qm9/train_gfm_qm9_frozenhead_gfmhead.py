#!/usr/bin/env python3
"""
Frozen-backbone GFM fine-tuning on QM9 with GFM-style head.
- Uses global label normalization
- Full sanity checks: param count, loader shape, forward shape
- Compatible with HydraGNN 2024
"""

import os, json, re, time, random, argparse, shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

import hydragnn
from hydragnn.utils.print_utils import setup_log, log
from hydragnn.utils.distributed import (
    setup_ddp, get_distributed_model, get_comm_size_and_rank
)
from hydragnn.utils import update_config
from hydragnn.models import create_model_config
import pdb
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fraction", required=True, choices=["1","5","10","25","50","100"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split_dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--label_stats", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def load_stats(path):
    d = json.load(open(path))
    return float(d["mean"]), float(d["std"])

def normalize(ds, mean, std):
    for d in ds:
        d.y = (d.y - mean) / std
    return ds

def replace_head_gfm_style(model):
    in_dim = model.heads_NN[0][0].in_features
    model.heads_NN = nn.ModuleList([
        nn.Sequential(
            nn.Linear(in_dim, 888),
            nn.ReLU(),
            nn.Linear(888, 888),
            nn.ReLU(),
            nn.Linear(888, 888),
            nn.ReLU(),
            nn.Linear(888, 1)
        )
    ])
    log(f"[HEAD] Replaced with GFM-style: {in_dim} → [888,888,888] → 1")
    log(f"[HEAD] Init head₀ last layer weight norm: {model.heads_NN[0][-1].weight.norm().item():.4f}")

def freeze_backbone(model):
    total, frozen = 0, 0
    for name, p in model.named_parameters():
        if not any(k in name.lower() for k in ("head", "classifier", "output")):
            p.requires_grad = False
            frozen += p.numel()
        total += p.numel()
    pct = 100 * frozen / total
    log(f"[SANITY] Frozen {frozen}/{total} params ({pct:.1f}%)")
    assert pct > 30, "Backbone not frozen enough"

def sanity_loader(loader, name):
    batch = next(iter(loader))
    B = int(batch.batch.max().item()) + 1
    assert batch.y.shape == (B, 1), f"{name}.y shape={batch.y.shape}, expected=({B},1)"
    log(f"[SANITY] {name} y.shape OK: {tuple(batch.y.shape)}")

def sanity_model_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[SANITY] Params: total={total:,}, trainable={trainable:,}")
    assert trainable > 0, "No trainable parameters"

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
    tag = f"qm9_gfmhead_frozen_{args.fraction}_{os.path.basename(args.split_dir)}_seed{args.seed}"
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
   
    # Load + normalize
    base = args.split_dir
    train_ds = normalize(torch.load(f"{base}/train_{args.fraction}.pt"), mean, std)
    val_ds   = normalize(torch.load(f"{base}/val.pt"), mean, std)
    test_ds  = normalize(torch.load(f"{base}/test.pt"), mean, std)
   
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        avg_y = torch.stack([d.y for d in ds]).mean().item()
        log(f"[CHECK] mean({name}.y) = {avg_y:.4f}")

    bs = cfg["NeuralNetwork"]["Training"]["batch_size"]
    class YFix:
        def __init__(self, loader): self.loader = loader
        def __iter__(self):
            for batch in self.loader:
                if batch.y.dim() == 1:
                    batch.y = batch.y.unsqueeze(1)
                yield batch
        def __len__(self): return len(self.loader)
        def __getattr__(self, name): return getattr(self.loader, name)

    train_loader = YFix(DataLoader(train_ds, batch_size=bs, shuffle=True))
    val_loader   = YFix(DataLoader(val_ds, batch_size=bs))
    test_loader  = YFix(DataLoader(test_ds, batch_size=bs))

    sanity_loader(train_loader, "train")
    sanity_loader(val_loader, "val")
    sanity_loader(test_loader, "test")

    update_config(cfg, train_loader, val_loader, test_loader)
    model = create_model_config(cfg["NeuralNetwork"], cfg["Verbosity"]["level"])

    replace_head_gfm_style(model)  
    freeze_backbone(model)         

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    bb = {k: v for k, v in state.items() if not k.startswith("heads_NN")}
    log(f"[LOAD] Loading {len(bb)} backbone keys")
    missing, _ = model.load_state_dict(bb, strict=False)
    if any("heads" not in k for k in missing):
        raise RuntimeError(f"[ERROR] Backbone mismatch: {missing}")
    log(f"[LOAD] Skipped head keys: {len(missing)}")

    sanity_model_params(model)
    model = get_distributed_model(model.to("cuda"), cfg["Verbosity"]["level"])

    with torch.no_grad():
        batch = next(iter(train_loader)).to("cuda")
        out = model(batch)
        if isinstance(out, (list, tuple)):
            out = torch.cat([o.reshape(-1, 1) for o in out], dim=1)
        assert out.shape == batch.y.shape, f"forward out {out.shape} != y {batch.y.shape}"
        log(f"[SANITY] forward OK: out.shape={tuple(out.shape)}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=5
    )
    writer = SummaryWriter(log_dir=f"./logs/{tag}")
    log(f"[TRAIN] Starting → tag={tag}")
    t0 = time.time()

    hydragnn.train.train_validate_test(
        model, optimizer,
        train_loader, val_loader, test_loader,
        writer, scheduler,
        cfg["NeuralNetwork"], tag, cfg["Verbosity"]["level"],
        create_plots=cfg.get("Visualization", {}).get("create_plots", True)
    )
    log(f"[DONE] Training finished in {time.time()-t0:.1f} sec")

    if get_comm_size_and_rank()[1] == 0:
        run_log = f"./logs/{tag}/run.log"
        best_ep = find_best_epoch(run_log)
        if best_ep is not None:
            src = f"./logs/{tag}/{tag}_epoch_{best_ep}.pk"
            dst = f"./logs/{tag}/best_{os.path.basename(args.split_dir)}.pt"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            log(f"[CHECKPOINT] Saved best epoch {best_ep} → {dst}")
        else:
            log("[WARN] Could not find best epoch")
