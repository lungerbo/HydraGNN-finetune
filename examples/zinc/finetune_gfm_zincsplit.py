import os
import json
import torch
import argparse
import hydragnn
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import torch.distributed as dist

# -------------------------
# Argument parsing
# -------------------------
parser = argparse.ArgumentParser(description="Fine-tune GFM on ZINC splits")
parser.add_argument("--split", type=int, required=True, help="Training split percentage (1, 3, 5, 10, 20, 100)")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to GFM checkpoint (.pk)")
parser.add_argument("--config", type=str, default="zinc.json", help="Path to config JSON file")
args = parser.parse_args()

# -------------------------
# Config and Setup
# -------------------------
with open(args.config, "r") as f:
    config = json.load(f)

verbosity = config["Verbosity"]["level"]

# Load datasets
train_set = torch.load(f"dataset/zinc_splits/zinc_train_{args.split}.pt")
val_set = torch.load("dataset/zinc_val.pt")
test_set = torch.load("dataset/zinc_test.pt")

train_loader, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
    train_set, val_set, test_set, config["NeuralNetwork"]["Training"]["batch_size"]
)

# Setup DDP (for CPU only)
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29500", rank=0, world_size=1)

config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)

# -------------------------
# Load GFM Model
# -------------------------
def load_gfm_model(config, checkpoint_path):
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"], strict=False)
    print(f"✅ Loaded GFM checkpoint: {checkpoint_path}")
    return model

model = load_gfm_model(config, args.checkpoint)
model = hydragnn.utils.get_distributed_model(model, verbosity)

# -------------------------
# Optimizer and Scheduler
# -------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)

log_name = f"finetune_gfm_zinc_{args.split}"
writer = hydragnn.utils.get_summary_writer(log_name)
hydragnn.utils.save_config(config, log_name)

# -------------------------
# Fine-tune GFM
# -------------------------
hydragnn.train.train_validate_test(
    model, optimizer, train_loader, val_loader, test_loader,
    writer, scheduler, config["NeuralNetwork"], log_name, verbosity,
    create_plots=config["Visualization"]["create_plots"]
)

# -------------------------
# Save Fine-tuned Model
# -------------------------
output_path = f"finetuned_gfm_zinc_{args.split}.pt"
torch.save({"model_state_dict": model.state_dict()}, output_path)
print(f"✅ Fine-tuned model saved to {output_path}")

# -------------------------
# Final Evaluation on Test Set
# -------------------------
model.eval()
y_true, y_pred = [], []

for batch in test_loader:
    batch = batch.to("cpu")
    with torch.no_grad():
        pred = model(batch)
    y_true.extend(batch.y.view(-1).tolist())
    if isinstance(pred, list):
        pred = torch.cat([p.view(-1) for p in pred])
    elif isinstance(pred, torch.Tensor) and pred.dim() > 1:
        pred = pred.view(-1)
    y_pred.extend(pred.tolist())

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
print(f"\n📊 Final Evaluation after Fine-tuning on {args.split}% split:")
print(f"✅ MAE: {mae:.6f}")
print(f"✅ R² : {r2:.6f}")

