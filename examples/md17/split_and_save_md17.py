import os
import torch
import random
import torch_geometric
from torch_geometric.nn import radius_graph

def md17_pre_transform(data):
    # Node features: [atomic_number, x, y, z]
    atomic_number = data.z.view(-1, 1).float()
    coordinates = data.pos.float()
    data.x = torch.cat([atomic_number, coordinates], dim=1)  # [num_nodes, 4]
    data.y = data.energy.view(1)
    # Ensure edge_index is set (radius graph, r=5.0 can be changed)
    data.edge_index = radius_graph(data.pos, r=5.0, batch=None, loop=False)
    # Optionally add edge_attr (distance as feature)
    row, col = data.edge_index
    edge_vec = data.pos[row] - data.pos[col]
    edge_length = torch.norm(edge_vec, dim=1, keepdim=True)
    data.edge_attr = edge_length
    return data

def split_dataset(dataset, perc_train=0.7, perc_val=0.15, seed=42):
    num_samples = len(dataset)
    indices = list(range(num_samples))
    random.seed(seed)
    random.shuffle(indices)
    n_train = int(perc_train * num_samples)
    n_val = int(perc_val * num_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]
    return train_set, val_set, test_set

def subsample_train(train_set, fraction, seed=42):
    num_samples = len(train_set)
    num_sub = max(1, int(fraction * num_samples))
    random.seed(seed)
    indices = random.sample(range(num_samples), num_sub)
    return [train_set[i] for i in indices]

def save_split(dataset, filename):
    torch.save(dataset, filename)

if __name__ == "__main__":
    # Patch for MD17 filename if needed
    torch_geometric.datasets.MD17.file_names["uracil"] = "md17_uracil.npz"
    dataset = torch_geometric.datasets.MD17(
        root="dataset/md17",
        name="uracil",
        pre_transform=md17_pre_transform,
        pre_filter=None,
    )
    perc_train, perc_val, seed = 0.7, 0.15, 42
    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
    out_dir = "md17_splits"
    os.makedirs(out_dir, exist_ok=True)
    train_set, val_set, test_set = split_dataset(dataset, perc_train, perc_val, seed)
    print(f"Full train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")
    save_split(train_set, os.path.join(out_dir, "train_full.pt"))
    save_split(val_set, os.path.join(out_dir, "val.pt"))
    save_split(test_set, os.path.join(out_dir, "test.pt"))
    for frac in fractions:
        frac_tag = str(int(frac * 100))
        sub_train = subsample_train(train_set, frac, seed=seed)
        save_split(sub_train, os.path.join(out_dir, f"train_{frac_tag}.pt"))
        print(f"Train fraction {frac}: {len(sub_train)} samples saved to train_{frac_tag}.pt")
    print(f"ALL splits saved in {out_dir}")
