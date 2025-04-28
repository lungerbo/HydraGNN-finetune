import os
import json
import torch
import argparse
import random
from torch_geometric.datasets import ZINC
import hydragnn

# -------------------------
# Argument parsing
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--split", type=int, required=True, help="Training split percentage (1, 3, 5, 10, 20, 100)")
parser.add_argument("--config", type=str, default="zinc.json", help="Path to model config")
args = parser.parse_args()

# -------------------------
# Environment setup
# -------------------------
if "SERIALIZED_DATA_PATH" not in os.environ:
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

with open(args.config, "r") as f:
    config = json.load(f)

verbosity = config["Verbosity"]["level"]
arch_config = config["NeuralNetwork"]["Architecture"]

world_size, world_rank = hydragnn.utils.distributed.setup_ddp()
log_name = f"zinc_split_{args.split}"
hydragnn.utils.setup_log(log_name)

# -------------------------
# Graph constructor
# -------------------------
compute_edges = hydragnn.preprocess.get_radius_graph_config(arch_config)

# Pre-transform function
def zinc_pre_transform(data):
    data.x = data.x.float()  # keep existing node features
    data.y = data.y.view(-1, 1)  # ensure target is [N,1]
    data = compute_edges(data)
    return data

# -------------------------
# Load dataset
# -------------------------
dataset = ZINC(root="dataset/zinc", subset=True, pre_transform=zinc_pre_transform)

# Shuffle + split
random.seed(42)
indices = list(range(len(dataset)))
random.shuffle(indices)

n_total = len(dataset)
n_test = int(n_total * 0.2)
n_val = int(n_total * 0.1)
n_train = n_total - n_val - n_test

test_indices = indices[:n_test]
val_indices = indices[n_test:n_test + n_val]
train_full_indices = indices[n_test + n_val:]

n_subset = int(len(train_full_indices) * args.split / 100)
train_indices = train_full_indices[:n_subset]

train_set = [dataset[i] for i in train_indices]
val_set = [dataset[i] for i in val_indices]
test_set = [dataset[i] for i in test_indices]

# -------------------------
# Save datasets
# -------------------------
# Save val/test once
os.makedirs("dataset", exist_ok=True)
torch.save(val_set, "dataset/zinc_val.pt")
torch.save(test_set, "dataset/zinc_test.pt")

# Save specific training split
os.makedirs("dataset/zinc_splits", exist_ok=True)
torch.save(train_set, f"dataset/zinc_splits/zinc_train_{args.split}.pt")

print(f"✅ Saved: {len(train_set)} training samples (split {args.split}%)")
print(f"✅ Saved: {len(val_set)} validation samples")
print(f"✅ Saved: {len(test_set)} test samples")
