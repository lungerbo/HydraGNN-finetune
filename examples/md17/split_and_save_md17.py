import os
import torch
import random
import torch_geometric
from torch_geometric.transforms import AddLaplacianEigenvectorPE
import hydragnn

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

def md17_pre_transform(data, compute_edges, transform):
    data.x = data.z.float().view(-1, 1)
    data.y = data.energy / len(data.x)
    data = compute_edges(data)
    data = transform(data)
    source_pe = data.pe[data.edge_index[0]]
    target_pe = data.pe[data.edge_index[1]]
    data.rel_pe = torch.abs(source_pe - target_pe)
    return data

if __name__ == "__main__":
    # Parameters
    molecule = "uracil"  # change if you want other MD17 molecules
    arch_config = {
        "radius": 7,
        "max_neighbours": 5,
        "pe_dim": 6
    }
    perc_train = 0.7
    perc_val = 0.15
    seed = 42
    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
    out_dir = f"md17_splits_{molecule}"
    os.makedirs(out_dir, exist_ok=True)

    compute_edges = hydragnn.preprocess.get_radius_graph_config(arch_config)
    transform = AddLaplacianEigenvectorPE(
        k=arch_config["pe_dim"], attr_name="pe", is_undirected=True
    )

    torch_geometric.datasets.MD17.file_names[molecule] = f"md17_{molecule}.npz"
    dataset = torch_geometric.datasets.MD17(
        root="dataset/md17",
        name=molecule,
        pre_transform=lambda data: md17_pre_transform(data, compute_edges, transform),
        pre_filter=None,  # No filter: use all data
    )

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

    print("All splits saved. To load: use torch.load(filename)")