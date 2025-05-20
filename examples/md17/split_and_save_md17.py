import os
import json
import torch
import random
import torch_geometric

import hydragnn

def md17_pre_transform(data):
    data.x = data.z.float().view(-1, 1)
    data.y = data.energy / len(data.x)
    data = compute_edges(data)
    return data

def split_dataset(dataset, perc_train=0.7, perc_val=0.15, seed=0):
    num_samples = len(dataset)
    indices = list(range(num_samples))
    random.seed(seed)
    random.shuffle(indices)
    n_train = int(perc_train * num_samples)
    n_val = int(perc_val * num_samples)
    n_test = num_samples - n_train - n_val
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]
    return train_set, val_set, test_set

def subsample_train(train_set, fraction, seed=0):
    num_samples = len(train_set)
    num_sub = max(1, int(fraction * num_samples))
    random.seed(seed)
    indices = random.sample(range(num_samples), num_sub)
    return [train_set[i] for i in indices]

def save_split(dataset, filename):
    torch.save(dataset, filename)

def load_split(filename):
    return torch.load(filename)

if __name__ == "__main__":
    # Load config
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "md17.json")
    with open(filename, "r") as f:
        config = json.load(f)
    perc_train = config["NeuralNetwork"]["Training"].get("perc_train", 0.7)
    perc_val = 0.15
    seed = 42
    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
    out_dir = "md17_splits"
    os.makedirs(out_dir, exist_ok=True)

    # Setup
    arch_config = config["NeuralNetwork"]["Architecture"]
    compute_edges = hydragnn.preprocess.get_radius_graph_config(arch_config)

    # Fix for MD17 dataset (Uracil)
    torch_geometric.datasets.MD17.file_names["uracil"] = "md17_uracil.npz"
    dataset = torch_geometric.datasets.MD17(
        root="dataset/md17",
        name="uracil",
        pre_transform=md17_pre_transform,
        pre_filter=None,  # Use full dataset
    )

    # Split
    train_set, val_set, test_set = split_dataset(dataset, perc_train=perc_train, perc_val=perc_val, seed=seed)
    print(f"Full train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")

    # Save full splits
    save_split(train_set, os.path.join(out_dir, "train_full.pt"))
    save_split(val_set, os.path.join(out_dir, "val.pt"))
    save_split(test_set, os.path.join(out_dir, "test.pt"))

    # Save fraction subsamples of the training set
    for frac in fractions:
        sub_train = subsample_train(train_set, frac, seed=seed)
        save_split(sub_train, os.path.join(out_dir, f"train_{int(frac*100)}.pt"))
        print(f"Train fraction {frac}: {len(sub_train)} samples saved to train_{int(frac*100)}.pt")
