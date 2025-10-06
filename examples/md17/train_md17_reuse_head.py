#!/usr/bin/env python3
"""
MD17 reuse-head fine-tuning with frozen GFM backbone.
- Loads GFM backbone and pretrained GFM head₀
- Only trains the head (backbone frozen)
- Global label normalization (using label_stats.json)
- Final checkpoint = best epoch
- Sanity checks: y shape, forward, and param counts
"""

import os, json, time, random, argparse, shutil, re
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

import hydragnn
from hydragnn.utils.print_utils import setup_log, log
from hydragnn.utils.distributed import setup_ddp, get_distributed_model, get_comm_size_and_rank
from hydragnn.utils import update_config
from hydragnn.models import create_model_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fraction", required=True, help="1|3|5|10|25|50|100")
    p.add_argument("--ckpt", required=True, help="Path to GFM checkpoint .pk")
    p.add_argument("--split_dir", required=True, help="Dir with train_*.pt, val.pt, test.pt, label_stats.json")
    p.add_argument("--config", required=True, help="HydraGNN JSON model config (head must match GFM)")
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


def Y_fix(loader):
    """Ensure batch.y is (B,1)"""
    class _Wrap:
        def __init__(self, ldr): self.loader = ldr
        def __iter__(self):
            for b in self.loader:
                if b.y.dim() == 1:
                    b.y = b.y.unsqueeze(1)
                yield b
        def __len__(self): return len(self.loader)
        def __getattr__(self, n): return getattr(self.loader, n)
    return _Wrap(loader)


def sanity_loader(loader, name):
    b = next(iter(loader))
    B = int(b.batch.max().item()) + 1 if hasattr(b, "batch") else b.y.shape[0]
    assert b.y.shape == (B, 1), f"{name}.y shape={b.y.shape}, expected=({B},1)"
    log(f"[SANITY] {name} y.shape OK: {tuple(b.y.shape)}")


def freeze_backbone_keep_head(model):
    """Freeze everything except head (graph_shared + heads_NN)."""
    total, trainable = 0, 0
    for n, p in model.named_parameters():
        # keep head trainable
        keep = ("graph_shared" in n) or ("heads_NN" in n)
        p.requires_grad = keep
        total += p.numel()
        if keep:
            trainable += p.numel()
    pct = 100.0 * trainable / total
    log(f"[SANITY] Trainable params (head only): {trainable:,}/{total:,} ({pct:.1f}%)")
    assert trainable > 0, "No head parameters left trainable!"


def patch_safe_forward_int_slice():
    """Cast outputs_total_dimensions to plain ints to avoid slicing errors."""
    try:
        from hydragnn.models.Base import Base as _Base
        _orig = _Base.forward
        def _safe(self, data, *a, **k):
            if hasattr(self, "outputs_total_dimensions"):
                self.outputs_total_dimensions = [int(x) for x in self.outputs_total_dimensions]
            return _orig(self, data, *a, **k)
        _Base.forward = _safe
        log("[PATCH] Safe forward: cast outputs_total_dimensions -> int")
    except Exception as e:
        log(f"[PATCH] Safe forward patch skipped: {e}")


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


if __name__ == "__main__":
    args = parse_args()
    tag = f"md17_reusehead_frozen_{args.fraction}_{os.path.basename(args.split_dir)}_seed{args.seed}"
    setup_log(tag)

    # seeds + device
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    os.environ["HYDRAGNN_DEVICE"] = "cuda"
    _, rank = setup_ddp()
    log(f"[DDP] Rank {rank}/?")

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
    log(f"Loaded datasets: {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test")

    # loaders (+ y-shape guard)
    bs = int(cfg["NeuralNetwork"]["Training"]["batch_size"])
    train_loader = Y_fix(DataLoader(train_ds, batch_size=bs, shuffle=True))
    val_loader   = Y_fix(DataLoader(val_ds, batch_size=bs))
    test_loader  = Y_fix(DataLoader(test_ds, batch_size=bs))
    sanity_loader(train_loader, "train"); sanity_loader(val_loader, "val"); sanity_loader(test_loader, "test")

    # reflect dataset dims into cfg
    update_config(cfg, train_loader, val_loader, test_loader)

    # build model
    patch_safe_forward_int_slice()
    model = create_model_config(cfg["NeuralNetwork"], cfg["Verbosity"]["level"])

    # load pretrained backbone + head from GFM
    ckpt  = torch.load(args.ckpt, map_location="cpu")
    state = strip_module(ckpt.get("model_state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    log(f"[LOAD] GFM checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")

    # freeze backbone, keep head trainable
    freeze_backbone_keep_head(model)

    # quick forward sanity
    model = get_distributed_model(model.to("cuda"), cfg["Verbosity"]["level"])
    with torch.no_grad():
        b = next(iter(train_loader)).to("cuda")
        out = model(b)
        if isinstance(out, (list, tuple)):
            out = torch.cat([o.reshape(-1, 1) for o in out], dim=1)
        assert out.shape == b.y.shape, f"forward out {out.shape} != y {b.y.shape}"
        log(f"[SANITY] forward OK: out.shape={tuple(out.shape)}")

    # optimize head only
    lr = float(cfg["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)

    # train
    writer = SummaryWriter(log_dir=f"./logs/{tag}")
    log(f"[TRAIN] Starting reuse-head training → tag={tag}")
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
