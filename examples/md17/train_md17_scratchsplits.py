import os
import json
import torch
from torch_geometric.loader import DataLoader
import hydragnn

# --- Set fraction ---
split_dir = "md17_splits"
train_fraction = "10"  # string: "1", "5", "10", "25", "50", "100"

# --- Load config ---
config_file = "config.json"
with open(config_file, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]

# --- Load splits ---
train_set = torch.load(os.path.join(split_dir, f"train_{train_fraction}.pt"))
val_set = torch.load(os.path.join(split_dir, "val.pt"))
test_set = torch.load(os.path.join(split_dir, "test.pt"))
print(f"Loaded split: train_{train_fraction}.pt ({len(train_set)}), val ({len(val_set)}), test ({len(test_set)})")

train_loader = DataLoader(train_set, batch_size=config["NeuralNetwork"]["Training"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config["NeuralNetwork"]["Training"]["batch_size"])
test_loader = DataLoader(test_set, batch_size=config["NeuralNetwork"]["Training"]["batch_size"])

os.environ.setdefault("SERIALIZED_DATA_PATH", os.getcwd())
world_size, world_rank = hydragnn.utils.distributed.setup_ddp()
log_name = f"md17_scratch_{train_fraction}"
hydragnn.utils.setup_log(log_name)

hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
model = hydragnn.models.create_model_config(config=config["NeuralNetwork"], verbosity=verbosity)
model = hydragnn.utils.get_distributed_model(model, verbosity)

learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
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
print(f"SCRATCH (energy): R2 Score: {r2:.4f}, MAE: {mae:.4f}")
