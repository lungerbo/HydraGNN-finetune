import os
import json
import torch
from torch_geometric.loader import DataLoader
import hydragnn
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

# ---------------------------
# User configuration section
# ---------------------------
split_dir = "qm9_splits"
train_fraction = "10"  # Use string: "1", "5", "10", etc.
checkpoint_path = "gfm_0.229.pk"  # GFM checkpoint path
config_path = "config.json"

# ---------------------------
# Load config and verbosity
# ---------------------------
with open(config_path, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]

# ---------------------------
# Load dataset splits
# ---------------------------
train_set = torch.load(os.path.join(split_dir, f"train_{train_fraction}.pt"))
val_set = torch.load(os.path.join(split_dir, "val.pt"))
test_set = torch.load(os.path.join(split_dir, "test.pt"))

# ---------------------------
# Create data loaders
# ---------------------------
batch_size = config["NeuralNetwork"]["Training"]["batch_size"]
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

#  DEBUG: Check input feature shape
sample_batch = next(iter(train_loader))
print(" Debug: Sample input x shape:", sample_batch.x.shape)
print(" Debug: Expected input_dim from config:", config["NeuralNetwork"]["Architecture"]["input_dim"])

# ---------------------------
# Setup HydraGNN environment
# ---------------------------
os.environ.setdefault("SERIALIZED_DATA_PATH", os.getcwd())
world_size, world_rank = hydragnn.utils.distributed.setup_ddp()
log_name = f"qm9_gfm_{train_fraction}"
hydragnn.utils.setup_log(log_name)

# ---------------------------
# Build model and load checkpoint
# ---------------------------
hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
model = hydragnn.models.create_model_config(config=config["NeuralNetwork"], verbosity=verbosity)
model = hydragnn.utils.get_distributed_model(model, verbosity)

def strip_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

if os.path.isfile(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    try:
        model.load_state_dict(state_dict, strict=False)
        print(f" Loaded GFM checkpoint from {checkpoint_path}")
    except RuntimeError as e:
        print("f Strict load failed, trying with prefix-stripped keys...", e)
        model.load_state_dict(strip_module_prefix(state_dict), strict=False)
else:
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
def print_first_weights(model, label):
    print(f"{label} first 5 weights:", next(model.parameters()).view(-1)[:5].tolist())

# 1. Print initial (random) weights
print_first_weights(model, "BEFORE loading checkpoint")

# 2. Load checkpoint as before
if os.path.isfile(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    print("Checkpoint keys:", list(ckpt.keys()))
    try:
        model.load_state_dict(state_dict, strict=False)
        print(f" Loaded GFM checkpoint from {checkpoint_path}")
    except RuntimeError as e:
        print(" Strict load failed, trying with prefix-stripped keys...", e)
        model.load_state_dict(strip_module_prefix(state_dict), strict=False)
else:
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

# 3. Print weights after loading
print_first_weights(model, "AFTER loading checkpoint")
# ---------------------------
# Optimizer and scheduler
# ---------------------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
)

# ---------------------------
# Writer and logging
# ---------------------------
writer = hydragnn.utils.get_summary_writer(log_name)
hydragnn.utils.save_config(config, log_name)

# ---------------------------
# Train/validate/test
# ---------------------------
hydragnn.train.train_validate_test(
    model, optimizer, train_loader, val_loader, test_loader,
    writer, scheduler, config["NeuralNetwork"], log_name, verbosity,
    create_plots=config["Visualization"]["create_plots"]
)

# ---------------------------
# Final test evaluation (R2/MAE)
# ---------------------------
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(model.device)
        pred = model(batch)
        # If model returns a tuple/list, take the first element
        graph_pred = pred[0] if isinstance(pred, (list, tuple)) else pred
        y_true.append(batch.y.cpu().numpy())
        y_pred.append(graph_pred.cpu().numpy())

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)
print(f" GFM FT (QM9): R² Score: {r2_score(y_true, y_pred):.4f}, MAE: {mean_absolute_error(y_true, y_pred):.4f}")
