import os, json

import torch
import torch_geometric
import random
import argparse
# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import hydragnn

import hydragnn
from sklearn.metrics import r2_score, mean_absolute_error

# ----- ARGUMENT PARSING -----
parser = argparse.ArgumentParser(description="Train QM9 with HydraGNN")
parser.add_argument("--split", type=int, default=100, help="Percentage of training dataset to use")
args = parser.parse_args()

num_samples = None  # Set a limit if needed

# ----- PREPROCESSING FUNCTION -----
def qm9_pre_transform(data):
    # Set descriptor as element type.
    data.x = data.z.float().view(-1, 1)
    # Only predict free energy (index 10 of 19 properties) for this run.
    data.y = data.y[:, 10] / len(data.x)
    graph_features_dim = [1]
    node_feature_dim = [1]
    return data

def qm9_pre_filter(data):
    return data.idx < num_samples if num_samples is not None else True

# ----- LOAD CONFIGURATION -----
filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm9.json")
with open(filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]

dataset = torch_geometric.datasets.QM9(
    root="dataset/qm9", pre_transform=qm9_pre_transform, pre_filter=qm9_pre_filter
)

# ----- DATASET SPLITTING -----
def create_splits(dataset, test_fraction=0.2, val_fraction=0.1, train_ratios=[0, 1, 3, 5, 10, 20, 50, 100]):
    n = len(dataset)
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)

    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)

    test_set = [dataset[i] for i in indices[:n_test]]
    val_set = [dataset[i] for i in indices[n_test:n_test + n_val]]
    train_set_full = [dataset[i] for i in indices[n_test + n_val:]]

    splits = {}
    for percent in train_ratios:
        n_train = int(len(train_set_full) * (percent / 100))
        train_set = train_set_full[:n_train]
        splits[percent] = (train_set, val_set, test_set)
        torch.save(train_set, f"dataset/qm9_train_{percent}.pt")
        torch.save(val_set, f"dataset/qm9_val.pt")
        torch.save(test_set, f"dataset/qm9_test.pt")
    return splits

# Create and save dataset splits
splits = create_splits(dataset)

# Load requested split
train_set = torch.load(f"dataset/qm9_train_{args.split}.pt")
val_set = torch.load("dataset/qm9_val.pt")
test_set = torch.load("dataset/qm9_test.pt")

# ----- DATA SPLIT DETAILS -----
total_samples = len(dataset)
train_samples = len(train_set)
val_samples = len(val_set)
test_samples = len(test_set)

print("\n\U0001F4CA Dataset Split Summary:")
print(f"Total QM9 samples: {total_samples}")
print(f"Training samples ({args.split}% split): {train_samples} ({100*train_samples/total_samples:.2f}%)")
print(f"Validation samples: {val_samples} ({100*val_samples/total_samples:.2f}%)")
print(f"Test samples: {test_samples} ({100*test_samples/total_samples:.2f}%)")

num_atoms_train = [len(data.x) for data in train_set]
num_atoms_val = [len(data.x) for data in val_set]
num_atoms_test = [len(data.x) for data in test_set]

print("\n\U0001F4CF Molecule Size Summary (Number of Atoms per Molecule):")
print(f"Train set: min={min(num_atoms_train)}, max={max(num_atoms_train)}, avg={sum(num_atoms_train)/len(num_atoms_train):.2f}")
print(f"Val set  : min={min(num_atoms_val)}, max={max(num_atoms_val)}, avg={sum(num_atoms_val)/len(num_atoms_val):.2f}")
print(f"Test set : min={min(num_atoms_test)}, max={max(num_atoms_test)}, avg={sum(num_atoms_test)/len(num_atoms_test):.2f}\n")

# ----- INITIALIZE HYDRAGNN -----
world_size, world_rank = hydragnn.utils.setup_ddp()
log_name = f"qm9_test_{args.split}"
hydragnn.utils.setup_log(log_name)

train_loader, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
    train_set, val_set, test_set, config["NeuralNetwork"]["Training"]["batch_size"])

config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)

model = hydragnn.models.create_model_config(config=config["NeuralNetwork"], verbosity=verbosity)
model = hydragnn.utils.get_distributed_model(model, verbosity)

learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)

writer = hydragnn.utils.get_summary_writer(log_name)
hydragnn.utils.save_config(config, log_name)

# ----- TRAINING -----
test_loss = hydragnn.train.train_validate_test(
    model, optimizer, train_loader, val_loader, test_loader,
    writer, scheduler, config["NeuralNetwork"], log_name, verbosity,
    create_plots=config["Visualization"]["create_plots"])

if args.split == 100:
    pretrained_path = f"pretrained_qm9_{args.split}.pt"
    torch.save({"model_state_dict": model.state_dict()}, pretrained_path)
    print(f"\u2705 Pretrained model saved at {pretrained_path}")

# ----- EVALUATION METRICS -----
y_true = []
y_pred = []

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

print(f"Final Accuracy on {args.split}% dataset training completed.")
print(f"R² Score: {r2:.4f}, MAE: {mae:.4f}")
