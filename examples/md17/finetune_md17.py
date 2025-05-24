import os
import json
import torch
import numpy as np
from collections import OrderedDict
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import hydragnn

# --- User Parameters ---
split_dir = "qm9_splits"
train_fraction = "100"
checkpoint_path = "gfm_0.229.pk"
config_file = "config.json"

# --- Load config ---
with open(config_file, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
batch_size = config["NeuralNetwork"]["Training"]["batch_size"]

# --- Load splits ---
train_set = torch.load(os.path.join(split_dir, f"train_{train_fraction}.pt"))
val_set = torch.load(os.path.join(split_dir, "val.pt"))
test_set = torch.load(os.path.join(split_dir, "test.pt"))
print(f"Loaded split: train_{train_fraction}.pt ({len(train_set)}), val ({len(val_set)}), test ({len(test_set)})")

# --- Linear baseline normalization (residual training) ---
all_splits = [train_set, val_set, test_set]
all_elements = sorted({int(z) for split in all_splits for g in split for z in g.z.tolist()})
element_to_idx = {z: i for i, z in enumerate(all_elements)}

def get_element_fractions(graph):
    counts = np.zeros(len(all_elements))
    for z in graph.z.tolist():
        counts[element_to_idx[int(z)]] += 1
    return counts / counts.sum()

X_train = np.stack([get_element_fractions(g) for g in train_set])
y_train = np.array([g.y.item() for g in train_set])
reg = LinearRegression().fit(X_train, y_train)

# Subtract baseline from training targets
for split in all_splits:
    for g in split:
        baseline = reg.predict(get_element_fractions(g).reshape(1, -1))[0]
        g.y = torch.tensor(g.y.item() - baseline, dtype=torch.float32)
print("Subtracted linear baseline from QM9 targets (residual training).")

# --- Dataloaders ---
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

# --- HydraGNN setup ---
os.environ.setdefault("SERIALIZED_DATA_PATH", os.getcwd())
world_size, world_rank = hydragnn.utils.distributed.setup_ddp()
log_name = f"qm9_gfm_{train_fraction}"
hydragnn.utils.setup_log(log_name)

hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
model = hydragnn.models.create_model_config(config=config["NeuralNetwork"], verbosity=verbosity)
model = hydragnn.utils.get_distributed_model(model, verbosity)

# --- Load GFM checkpoint ---
def strip_module_prefix(state_dict):
    return OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state_dict.items())

if os.path.isfile(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cuda")
    state_dict = ckpt.get("model_state_dict", ckpt)
    try:
        load_result = model.load_state_dict(state_dict, strict=False)
    except RuntimeError:
        load_result = model.load_state_dict(strip_module_prefix(state_dict), strict=False)
    print(f"Loaded GFM checkpoint from {checkpoint_path}")
    print(f"Missing keys: {load_result.missing_keys}")
    print(f"Unexpected keys: {load_result.unexpected_keys}")
else:
    raise FileNotFoundError(f"No GFM checkpoint found at {checkpoint_path}")

# --- Optimizer & Trainer ---
optimizer = torch.optim.AdamW(model.parameters(), lr=config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)
writer = hydragnn.utils.get_summary_writer(log_name)
hydragnn.utils.save_config(config, log_name)

hydragnn.train.train_validate_test(
    model,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    writer,
    scheduler,
    config["NeuralNetwork"],
    log_name,
    verbosity,
    create_plots=config["Visualization"]["create_plots"],
)

# --- Evaluation (add baseline back) ---
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(model.device)
        pred = model(batch)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        y_pred.append(pred.view(-1).cpu().numpy())
        y_true.append(batch.y.view(-1).cpu().numpy())

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

# Add baseline back to predictions
X_test = np.stack([get_element_fractions(g) for g in test_set])
baseline_test = reg.predict(X_test)

y_true_total = y_true + baseline_test
y_pred_total = y_pred + baseline_test

print(f"GFM FT (QM9, total energy): R² = {r2_score(y_true_total, y_pred_total):.4f}, MAE = {mean_absolute_error(y_true_total, y_pred_total):.4f}")

torch.distributed.destroy_process_group()

