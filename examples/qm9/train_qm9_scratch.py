import os
import json
import torch
from torch_geometric.loader import DataLoader
import hydragnn
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

# Choose which training subset size to load
split_dir = "qm9_splits"
train_fraction = "10"  # Use string: "1", "5", "10", etc.

# Load config
with open("config.json", "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]

# Load dataset splits
train_set = torch.load(os.path.join(split_dir, f"train_{train_fraction}.pt"))
val_set = torch.load(os.path.join(split_dir, "val.pt"))
test_set = torch.load(os.path.join(split_dir, "test.pt"))

# Create data loaders
train_loader = DataLoader(train_set, batch_size=config["NeuralNetwork"]["Training"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config["NeuralNetwork"]["Training"]["batch_size"])
test_loader = DataLoader(test_set, batch_size=config["NeuralNetwork"]["Training"]["batch_size"])

#  DEBUG: Check input feature shape
sample_batch = next(iter(train_loader))
print(" Debug: Sample input x shape:", sample_batch.x.shape)
print(" Debug: Expected input_dim from config:", config["NeuralNetwork"]["Architecture"]["input_dim"])

# Set environment for HydraGNN
os.environ.setdefault("SERIALIZED_DATA_PATH", os.getcwd())
world_size, world_rank = hydragnn.utils.distributed.setup_ddp()
log_name = f"qm9_scratch_{train_fraction}"
hydragnn.utils.setup_log(log_name)

# Build model
hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
model = hydragnn.models.create_model_config(config=config["NeuralNetwork"], verbosity=verbosity)
model = hydragnn.utils.get_distributed_model(model, verbosity)

# Optimizer/scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)

# Writer + logging
writer = hydragnn.utils.get_summary_writer(log_name)
hydragnn.utils.save_config(config, log_name)

# Train and evaluate
hydragnn.train.train_validate_test(
    model, optimizer, train_loader, val_loader, test_loader,
    writer, scheduler, config["NeuralNetwork"], log_name, verbosity,
    create_plots=config["Visualization"]["create_plots"]
)

# Final test evaluation
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(model.device)
        pred = model(batch)
        graph_pred = pred[0] if isinstance(pred, (list, tuple)) else pred
        y_true.append(batch.y.cpu().numpy())
        y_pred.append(graph_pred.cpu().numpy())

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)
print(f" SCRATCH (QM9): R² Score: {r2_score(y_true, y_pred):.4f}, MAE: {mean_absolute_error(y_true, y_pred):.4f}")
