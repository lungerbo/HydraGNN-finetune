import os
import json
import torch
import torch_geometric
from collections import OrderedDict

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

# ---- CONFIG LOAD AND CHECK ----
config_file = "config.json"
print("Config absolute path:", os.path.abspath(config_file))
with open(config_file, "r") as f:
    config = json.load(f)
print("Config loaded:")
print(json.dumps(config, indent=2))

# Patch Variables_of_interest["type"] to a list if not already (fixes 'g' bug)
voi = config["NeuralNetwork"].get("Variables_of_interest", {})
if isinstance(voi.get("type"), str):
    voi["type"] = [voi["type"]]
    config["NeuralNetwork"]["Variables_of_interest"] = voi

if "Variables_of_interest" not in config["NeuralNetwork"]:
    raise KeyError("Your config is missing the 'Variables_of_interest' section under 'NeuralNetwork'! Please update your config.json.")
else:
    print("'Variables_of_interest' is present in config['NeuralNetwork'].")

voi = config["NeuralNetwork"]["Variables_of_interest"]
for key in ["type", "names", "dim"]:
    if key not in voi:
        raise KeyError(f"'Variables_of_interest' is missing the '{key}' key!")
print("'Variables_of_interest' has required keys.")

import hydragnn

arch_config = config["NeuralNetwork"]["Architecture"]
EDGE_DIM = arch_config.get("edge_dim", 10)

def compute_edges(data):
    if not hasattr(data, "edge_index") or data.edge_index is None:
        from torch_geometric.nn import radius_graph
        data.edge_index = radius_graph(data.pos, r=arch_config.get("radius", 5.0), batch=None, loop=False)
    row, col = data.edge_index
    edge_vec = data.pos[row] - data.pos[col]
    edge_length = torch.norm(edge_vec, dim=1, keepdim=True)
    data.edge_attr = edge_length
    return data

def gfm_pre_transform(data):
    atomic_number = data.z.view(-1, 1).float()
    coordinates = data.pos.float()
    data.x = torch.cat([atomic_number, coordinates], dim=1)
    # For ENERGY-ONLY: set data.y as a tensor
    data.y = data.energy.view(1)
    num_nodes = data.num_nodes
    data.y_loc = torch.tensor([[0, 1, 1 + num_nodes * 3]])
    data = compute_edges(data)
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        n_edgefeat = data.edge_attr.shape[1]
        if n_edgefeat < EDGE_DIM:
            pad = torch.zeros(data.edge_attr.shape[0], EDGE_DIM - n_edgefeat)
            data.edge_attr = torch.cat([data.edge_attr, pad], dim=1)
        elif n_edgefeat > EDGE_DIM:
            data.edge_attr = data.edge_attr[:, :EDGE_DIM]
    else:
        if hasattr(data, "edge_index") and data.edge_index is not None:
            n_edges = data.edge_index.shape[1]
            data.edge_attr = torch.zeros((n_edges, EDGE_DIM))
    return data

os.environ.setdefault("SERIALIZED_DATA_PATH", os.getcwd())
verbosity = config["Verbosity"]["level"]
world_size, world_rank = hydragnn.utils.distributed.setup_ddp()
log_name = "md17_gfm_head_retrain"
hydragnn.utils.setup_log(log_name)

# Patch for MD17 file name if needed
torch_geometric.datasets.MD17.file_names["uracil"] = "md17_uracil.npz"

processed_dir = os.path.join("dataset", "md17", "processed")
if os.path.exists(processed_dir):
    import shutil
    shutil.rmtree(processed_dir)

dataset = torch_geometric.datasets.MD17(
    root="dataset/md17",
    name="uracil",
    pre_transform=gfm_pre_transform,
    pre_filter=None,
)

perc_train = config["NeuralNetwork"]["Training"]["perc_train"]
train, val, test = hydragnn.preprocess.split_dataset(dataset, perc_train, False)
train_loader, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
    train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
)

print("Train/Val/Test splits:", len(train), len(val), len(test))
print("Batch size:", config["NeuralNetwork"]["Training"]["batch_size"])

# --- Pass the FULL config to update_config ---
hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
print("Config after update_config:")
print(json.dumps(config, indent=2))
if "Variables_of_interest" not in config["NeuralNetwork"]:
    raise KeyError("After update_config, 'Variables_of_interest' is missing under 'NeuralNetwork'! Something overwrote your config.")

model = hydragnn.models.create_model_config(
    config=config["NeuralNetwork"],
    verbosity=verbosity,
)
model = hydragnn.utils.get_distributed_model(model, verbosity)

def strip_module_prefix(state_dict):
    """Remove 'module.' prefix from keys, if present."""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith("module.") else k
        new_state_dict[new_key] = v
    return new_state_dict

# === Load GFM checkpoint with prefix strip ===
checkpoint_path = "gfm_0.229.pk"
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Always strip 'module.' prefix if present
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded GFM checkpoint from {checkpoint_path} (with prefix strip)")

learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
)

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

from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(model.device)
        pred = model(batch)
        if isinstance(pred, (list, tuple)) and len(pred) > 1:
            graph_pred = pred[0]
        else:
            graph_pred = pred
        y_true.append(batch.y.cpu().numpy())
        y_pred.append(graph_pred.cpu().numpy())
y_true = np.concatenate(y_true, axis=0)
y_pred = np.concatenate(y_pred, axis=0)

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
print(f"GFM Head Retrain (energy): R2 Score: {r2:.4f}, MAE: {mae:.4f}")
