#!/usr/bin/env python3
"""
Train MD17 (energy-only) with a GFM-matching head — QM9-style pipeline.

- Reads label_stats.json (train-only mean/std)
- Normalizes y on {train,val,test}
- Builds model from JSON (head must match GFM: 50 -> 888 -> 888 -> 888 -> 1)
- Loads GFM checkpoint (strict=False)
- Robust forward guard (casts any non-int head dims)
- Trains and saves best epoch like QM9
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
    p.add_argument("--ckpt", required=True, help="GFM checkpoint .pk (full head match)")
    p.add_argument("--split_dir", required=True, help="dir with train_*.pt, val.pt, test.pt, label_stats.json")
    p.add_argument("--config", required=True, help="HydraGNN JSON (matches GFM head)")
    p.add_argument("--label_stats", required=True, help="label_stats.json with {mean,std} (train-only)")
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


def install_forward_guard():
    """Wrap Base.forward once; on the specific TypeError, coerce dims to ints and retry."""
    from hydragnn.models.Base import Base as _Base
    if getattr(_Base, "_wrapped_intslice_guard", False):
        return
    _Base._wrapped_intslice_guard = True
    _orig_forward = _Base.forward

    def _safe_forward(self, data, *a, **k):
        try:
            return _orig_forward(self, data, *a, **k)
        except TypeError as e:
            if "slice indices must be integers" not in str(e):
                raise
            for attr in ("outputs_total_dimensions", "outputs_dimensions_nodes", "outputs_dimensions_graph"):
                if hasattr(self, attr):
                    seq = getattr(self, attr)
                    new = []
                    for x in seq:
                        try:
                            new.append(int(x))
                        except Exception:
                            try:
                                new.append(int(x.item()))
                            except Exception:
                                new.append(1)
                    setattr(self, attr, new)
            return _orig_forward(self, data, *a, **k)

    _Base.forward = _safe_forward


if __name__ == "__main__":
    args = parse_args()
    tag = f"md17_gfm_full_{args.fraction}_{os.path.basename(args.split_dir)}_seed{args.seed}"
    setup_log(tag)

    # seeds + device/DDP
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    os.environ["HYDRAGNN_DEVICE"] = "cuda"
    world_size, rank = setup_ddp()
    log(f"[DDP] Rank {rank}/{world_size}")

    # config + stats
    cfg = json.load(open(args.config))
    mean, std = load_stats(args.label_stats)
    log(f"[STATS] mean={mean:.6f}, std={std:.6f}")

    # datasets: energy-only (y.shape == [1])
    base = args.split_dir
    train_file = f"train_{args.fraction}.pt" if args.fraction != "100" else "train_full.pt"
    train_ds = normalize(torch.load(os.path.join(base, train_file)), mean, std)
    val_ds   = normalize(torch.load(os.path.join(base, "val.pt")),  mean, std)
    test_ds  = normalize(torch.load(os.path.join(base, "test.pt")), mean, std)
    log(f"✅ Loaded datasets: {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test")

    # loaders + shape-driven config update
    bs = int(cfg["NeuralNetwork"]["Training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=bs)
    test_loader  = DataLoader(test_ds, batch_size=bs)
    update_config(cfg, train_loader, val_loader, test_loader)

    # model
    model = create_model_config(cfg["NeuralNetwork"], cfg["Verbosity"]["level"]).to("cuda")

    # robust forward guard (PyTorch 2.4 slice bug path)
    install_forward_guard()

    # inspect head shapes (sanity)
    shared0 = tuple(model.graph_shared[0].weight.shape[::-1])  # (in,out) as repr
    last    = tuple(model.heads_NN[0][-1].weight.shape)         # (1, 888)
    log(f"[HEAD] shared0={shared0} last={last}")

    # snapshot head₀ BEFORE loading ckpt
    head0_init = {k: v.detach().clone() for k, v in model.heads_NN[0].state_dict().items()}

    # load checkpoint
    ckpt  = torch.load(args.ckpt, map_location="cpu")
    state = strip_module(ckpt.get("model_state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    log(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}")

    # head movement (after vs before)
    head0_loaded = {k: v.detach().clone() for k, v in model.heads_NN[0].state_dict().items()}
    max_diff = max((head0_loaded[k] - head0_init[k]).abs().max().item() for k in head0_init)
    log(f"[SANITY] max |Δ head₀| = {max_diff:.3e}")

    # all params trainable?
    n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    log(f"[SANITY] Total params: {sum(p.numel() for p in model.parameters())}, frozen={n_frozen}")
    assert n_frozen == 0

    # DDP wrap
    model = get_distributed_model(model, cfg["Verbosity"]["level"])

    # opt/sched
    lr = float(cfg["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
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
            log(f"[CHECKPOINT] Best model saved → {dst}")
        else:
            log("[WARN] Could not find best epoch")
