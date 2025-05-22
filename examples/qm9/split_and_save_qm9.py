import os
import torch
import random
from torch_geometric.datasets import QM9
from torch_geometric.nn import radius_graph

def qm9_pre_transform(data):
    # Node features: [atomic_number, x, y, z]
    atomic_number = data.z.view(-1, 1).float()
    coordinates = data.pos.float()
    data.x = torch.cat([atomic_number, coordinates], dim=1)  # shape: [num_nodes, 4]

    # Label: free energy / U0 normalized by atom count
    data.y = data.y[:, 10] / len(data.x)

    # Edge index and edge length
    edge_index = radius_graph(coordinates, r=5.0, loop=False)
    row, col = edge_index
    edge_vec = coordinates[row] - coordinates[col]
    edge_length = torch.norm(edge_vec, dim=1, keepdim=True)
    data.edge_index = edge_index
    data.edge_attr = edge_length

    return data

def split_dataset(dataset, perc_train=0.7, perc_val=0.15, seed=42):
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    n_train = int(perc_train * len(dataset))
    n_val = int(perc_val * len(dataset))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    return [dataset[i] for i in train_idx], [dataset[i] for i in val_idx], [dataset[i] for i in test_idx]

def subsample_train(train_set, fraction, seed=42):
    n = max(1, int(len(train_set) * fraction))
    random.seed(seed)
    indices = random.sample(range(len(train_set)), n)
    return [train_set[i] for i in indices]

def save_split(dataset, path):
    torch.save(dataset, path)

if __name__ == "__main__":
    if os.path.exists("dataset/qm9/processed"):
        print("  Removing old processed data to apply updated pre_transform...")
        import shutil
        shutil.rmtree("dataset/qm9/processed")

    dataset = QM9(root="dataset/qm9", pre_transform=qm9_pre_transform)
    train_set, val_set, test_set = split_dataset(dataset, perc_train=0.7, perc_val=0.15, seed=42)

    os.makedirs("qm9_splits", exist_ok=True)
    save_split(train_set, "qm9_splits/train_full.pt")
    save_split(val_set, "qm9_splits/val.pt")
    save_split(test_set, "qm9_splits/test.pt")

    for frac in [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]:
        tag = str(int(frac * 100))
        sub = subsample_train(train_set, frac)
        save_split(sub, f"qm9_splits/train_{tag}.pt")
        print(f" Saved train_{tag}.pt ({len(sub)} samples)")

    print(" All QM9 splits written to qm9_splits/")
