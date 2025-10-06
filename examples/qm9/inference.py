#!/usr/bin/env python3
"""
inference_qm9_gfmhead_frozen.py

Performs inference for single-property QM9 with:
- GFM backbone + 3-layer GFM-style head
- Proper checkpoint loading
- Label denormalization
- MAE, RMSE, R² reporting
- True-vs-predicted scatter plot
"""

import os, json, argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader
import pdb
from hydragnn.utils.config_utils import update_config
from hydragnn.utils.distributed import setup_ddp, get_device
from hydragnn.models.create import create_model_config

def strip_module_keys(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}
def normalize(ds, mean, std):
    for d in ds:
        d.y = (d.y - mean) / std
    return ds
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split_dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--label_stats", required=True)
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    setup_ddp()

    # Load test data and label stats
    test_data = torch.load(os.path.join(args.split_dir, "test.pt"))
    with open(args.label_stats, "r") as f:
        stats = json.load(f)
    mean, std = float(stats["mean"]), float(stats["std"])
    test_data  = normalize(test_data, mean, std)
    print(f"[INFO] Loaded {len(test_data)} test graphs")
    print(f"[INFO] Label mean={mean:.4f}, std={std:.4f}")

    # Build loader and update config
    loader = DataLoader(test_data, batch_size=args.batch_size)
    full_cfg = json.load(open(args.config))
    update_config(full_cfg, loader, loader, loader)
    model = create_model_config(full_cfg["NeuralNetwork"], verbosity=0).to(get_device())

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = strip_module_keys(ckpt.get("model_state_dict", ckpt))
    missing = model.load_state_dict(state, strict=False)[0]
    skipped = [k for k in missing if "heads" not in k]
    assert not skipped, f"Missing backbone weights: {skipped}"
    print(f"[INFO] Loaded model with {len(missing)} head keys skipped")

    # Run inference
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(get_device())
            out = model(batch)
            out = out[0] if isinstance(out, (tuple, list)) else out
            y_pred = out.view(-1).cpu().numpy()
            y_true = batch.y.view(-1).cpu().numpy()

            preds.append(y_pred)
            targets.append(y_true)

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    print(np.mean(preds), np.std(preds), np.min(preds), np.max(preds))
    print(np.mean(targets), np.std(targets), np.min(targets), np.max(targets))
    
    
    preds_denorm = preds * std + mean
    targets_denorm = targets * std + mean

    # Compute metrics
    mae_val = mean_absolute_error(targets_denorm, preds_denorm)
    rmse_val = np.sqrt(mean_squared_error(targets_denorm, preds_denorm))
    r2_val = r2_score(targets_denorm, preds_denorm)
    print(f"[METRICS] MAE={mae_val:.4f}  RMSE={rmse_val:.4f}  R²={r2_val:.8f}")

    # Plot
    plt.figure(figsize=(6,6))
    plt.scatter(targets_denorm, preds_denorm, alpha=0.5, s=8)
    lims = [min(targets_denorm.min(), preds_denorm.min()),
            max(targets_denorm.max(), preds_denorm.max())]
    plt.plot(lims, lims, "--", color="gray")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{os.path.basename(args.split_dir)}")
    plt.tight_layout()
    png_name = os.path.splitext(os.path.basename(args.ckpt))[0] + "_scatter.png"
    plt.savefig(png_name, dpi=300)
    print(f"[PLOT] Saved to {png_name}")

    # DDP cleanup
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
