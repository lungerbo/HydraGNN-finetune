#!/usr/bin/env python3
"""
MD17: head-from-scratch, frozen GFM backbone.
- Energy-only targets (shape [B,1])
- Global label normalization
- Replace head with GFM-style MLP (scratch init)
- Load ONLY backbone from GFM ckpt; freeze backbone + graph_shared
- Train head only
- Forward-cast guard for PyTorch 2.4 slicing
"""

import os, json, re, time, random, argparse, shutil
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import hydragnn
from hydragnn.utils.print_utils import setup_log, log
from hydragnn.utils.distributed import setup_ddp, get_distributed_model, get_comm_size_and_rank
from hydragnn.utils import update_config
from hydragnn.models import create_model_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fraction", required=True, choices=["1","3","5","10","25","50","100"])
    p.add_argument("--ckpt", required=True, help="Path to GFM checkpoint (.pk)")
    p.add_argument("--split_dir", required=True, help="Dir with train_*.pt, val.pt, test.pt, label_stats.json")
    p.add_argument("--config", required=True, help="HydraGNN JSON config (EGNN 576→shared 50→head 888s)")
    p.add_argument("--label_stats", required=True, help="label_stats.json with {mean,std}")
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
    if not os.path.exists(logfile):
        return None
    for line in open(logfile):
        m = re.search(r"epoch (\d+).*?validation MAE: ([\d\.eE+-]+)", line)
        if m:
            ep, val = int(m.group(1)), float(m.group(2))
            if val < best_val:
                best_val, best_ep = val, ep
    return best_ep


def strip_module(sd):
    return {k.replace("module.", ""): v for k, v in sd.items()}


def replace_head_gfm_style(model):
    # Input to per-head MLP comes from the *shared* stack output (usually 50)
    try:
        in_dim = model.heads_NN[0][0].in_features
    except Exception:
        in_dim = model.graph_shared[-1].out_features
    model.heads_NN = nn.ModuleList([
        nn.Sequential(
            nn.Linear(in_dim, 888), nn.ReLU(),
            nn.Linear(888, 888),    nn.ReLU(),
            nn.Linear(888, 888),    nn.ReLU(),
            nn.Linear(888, 1),
        )
    ])
    log(f"[HEAD] scratch GFM-style head: {in_dim} → 888 → 888 → 888 → 1")


def freeze_backbone(model):
    total, frozen = 0, 0
    for name, p in model.named_parameters():
        # Only train the new head; freeze everything else (incl. graph_shared/backbone conv)
        if "heads_NN" not in name:
            p.requires_grad = False
            frozen += p.numel()
        total += p.numel()
    pct = 100.0 * frozen / max(1, total)
    log(f"[SANITY] Frozen {frozen}/{total} params ({pct:.1f}%)")
    assert pct > 30, "Backbone not frozen enough"


def sanity_loader(loader, name):
    b = next(iter(loader))
    B = int(b.batch.max().item()) + 1
    assert b.y.shape == (B, 1), f"{name}.y shape={tuple(b.y.shape)} expected ({B},1)"
    log(f"[SANITY] {name} y.shape OK: {tuple(b.y.shape)}")


def sanity_model_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[SANITY] Params: total={total:,}, trainable={trainable:,}")
    assert trainable > 0, "No trainable parameters"


if __name__ == "__main__":
    args = parse_args()
    tag = f"md17_headscratch_frozen_{args.fraction}_{os.path.basename(args.split_dir)}_seed{args.seed}"
    setup_log(tag)

    # seeds + device
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    os.environ["HYDRAGNN_DEVICE"] = "cuda"
    world_size, rank = setup_ddp()
    log(f"[DDP] Rank {rank}/{world_size}")

    # cast-guard for PyTorch 2.4 slicing in Base.forward
    try:
        from hydragnn.models.Base import Base as _Base
        _orig_fwd = _Base.forward
        def _safe_forward(self, data, *a, **k):
            if hasattr(self, "outputs_total_dimensions"):
                self.outputs_total_dimensions = [int(x) for x in self.outputs_total_dimensions]
            return _orig_fwd(self, data, *a, **k)
        _Base.forward = _safe_forward
    except Exception:
        pass

    # config + stats
    cfg = json.load(open(args.config))
    mean, std = load_stats(args.label_stats)
    log(f"[STATS] mean={mean:.6f}, std={std:.6f}")

    # datasets (energy-only)
    base = args.split_dir
    train_file = f"train_{args.fraction}.pt" if args.fraction != "100" else "train_full.pt"
    train_ds = normalize(torch.load(os.path.join(base, train_file)), mean, std)
    val_ds   = normalize(torch.load(os.path.join(base, "val.pt")),  mean, std)
    test_ds  = normalize(torch.load(os.path.join(base, "test.pt")), mean, std)
    log(f" Loaded datasets: {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test")

    # loaders (+ y shape fix)
    bs = int(cfg["NeuralNetwork"]["Training"]["batch_size"])
    class YFix:
        def __init__(self, loader): self.loader = loader
        def __iter__(self):
            for batch in self.loader:
                if batch.y.dim() == 1:
                    batch.y = batch.y.unsqueeze(1)
                yield batch
        def __len__(self): return len(self.loader)
        def __getattr__(self, n): return getattr(self.loader, n)

    train_loader = YFix(DataLoader(train_ds, batch_size=bs, shuffle=True))
    val_loader   = YFix(DataLoader(val_ds,   batch_size=bs))
    test_loader  = YFix(DataLoader(test_ds,  batch_size=bs))

    sanity_loader(train_loader, "train")
    sanity_loader(val_loader,   "val")
    sanity_loader(test_loader,  "test")

    # infer dims → model
    update_config(cfg, train_loader, val_loader, test_loader)
    model = create_model_config(cfg["NeuralNetwork"], cfg["Verbosity"]["level"])

    # replace head with scratch GFM-style MLP
    replace_head_gfm_style(model)

    # load ONLY backbone from ckpt
    ckpt  = torch.load(args.ckpt, map_location="cpu")
    state = strip_module(ckpt.get("model_state_dict", ckpt))
    backbone_only = {k: v for k, v in state.items() if not k.startswith("heads_NN")}
    missing, unexpected = model.load_state_dict(backbone_only, strict=False)
    log(f"[LOAD] backbone keys loaded={len(backbone_only)} | missing={len(missing)} | unexpected={len(unexpected)}")
    # OK if missing are all head keys; error if backbone tensors are missing
    bad = [k for k in missing if not k.startswith("heads_NN")]
    if bad:
        raise RuntimeError(f"[ERROR] Backbone mismatch on: {bad[:5]} ...")

    # freeze backbone (train head only)
    freeze_backbone(model)
    sanity_model_params(model)

    # DDP wrap + quick forward sanity
    model = get_distributed_model(model.to("cuda"), cfg["Verbosity"]["level"])
    with torch.no_grad():
        batch = next(iter(train_loader)).to("cuda")
        out = model(batch)
        if isinstance(out, (list, tuple)):
            out = torch.cat([o.reshape(-1, 1) for o in out], dim=1)
        assert out.shape == batch.y.shape, f"forward out {tuple(out.shape)} != y {tuple(batch.y.shape)}"
        log(f"[SANITY] forward OK: out.shape={tuple(out.shape)}")

    # optimizer/scheduler on trainable (head) only
    lr = float(cfg["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)

    # train
    writer = SummaryWriter(log_dir=f"./logs/{tag}")
    log(f"[TRAIN] Starting → tag={tag}")
    t0 = time.time()
    hydragnn.train.train_validate_test(
        model, optimizer, train_loader, val_loader, test_loader,
        writer, scheduler,
        cfg["NeuralNetwork"], tag, cfg["Verbosity"]["level"],
        create_plots=cfg.get("Visualization", {}).get("create_plots", True),
    )
    log(f"[DONE] Training completed in {time.time()-t0:.1f} sec")

    # save best
    if get_comm_size_and_rank()[1] == 0:
        run_log = f"./logs/{tag}/run.log"
        best_ep = find_best_epoch(run_log)
        if best_ep is not None:
            src = f"./logs/{tag}/{tag}_epoch_{best_ep}.pk"
            dst = f"./logs/{tag}/best_{os.path.basename(args.split_dir)}.pt"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            log(f"[CHECKPOINT] Saved best model → {dst}")
        else:
            log("[WARN] Could not find best epoch")
