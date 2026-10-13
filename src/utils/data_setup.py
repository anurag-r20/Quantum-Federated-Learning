"""
Contains functionality for creating PyTorch DataLoaders for image classification data,
plus a cached grayscale loader and client partitioning for the MRI privacy-attack experiments.
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count()

# Normalization values for the different datasets
NORMALIZE_DICT = {
    'cifar': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'MRI': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'mnist': dict(mean=(0.1307,), std=(0.3081,)),
}

MRI_CLASSES = ('glioma', 'meningioma', 'notumor', 'pituitary')
# Grayscale MRI tensors used by the attack pipeline live in [-1, 1]: x = (pixel / 255 - 0.5) / 0.5
MRI_GRAY_MEAN, MRI_GRAY_STD = 0.5, 0.5


def non_iid_mri_indices(targets, class_to_idx, num_clients=3):
    """
    Non-IID MRI split: client i gets every image of tumour class i (glioma, meningioma,
    pituitary) plus an equal share of the "no tumour" class.
    """
    if num_clients != 3:
        raise ValueError(f"Non-IID MRI partitioning requires exactly 3 clients, got {num_clients}")
    targets = np.asarray(targets)
    tumour_classes = [class_to_idx[name] for name in ('glioma', 'meningioma', 'pituitary')]
    no_tumour = np.flatnonzero(targets == class_to_idx['notumor'])
    shares = np.array_split(no_tumour, num_clients)
    return [np.concatenate([np.flatnonzero(targets == c), share]) for c, share in zip(tumour_classes, shares)]


def partition_indices(targets, num_clients, strategy='iid', seed=0, class_to_idx=None):
    """
    Split sample indices between clients.

    :param targets: labels of the samples
    :param strategy: 'iid' (random equal split) or 'non_iid' (class-specific MRI split)
    :param class_to_idx: class name -> label mapping (required for 'non_iid')
    :return: list of index arrays, one per client
    """
    if strategy == 'iid':
        permutation = np.random.default_rng(seed).permutation(len(targets))
        return np.array_split(permutation, num_clients)
    if strategy == 'non_iid':
        return non_iid_mri_indices(targets, class_to_idx or {c: i for i, c in enumerate(MRI_CLASSES)}, num_clients)
    raise ValueError(f"Unknown partition_strategy: {strategy}. Use 'iid' or 'non_iid'")


def split_data_client(dataset, num_clients, seed, partition_strategy='iid', dataset_type='MRI'):
    """
    This function is used to split the dataset between the clients.

    Args:
        dataset: PyTorch dataset to split (ImageFolder for 'non_iid')
        num_clients: Number of clients
        seed: Random seed for reproducibility
        partition_strategy: 'iid' (balanced) or 'non_iid' (class-specific, MRI only)
        dataset_type: Type of dataset (Non-IID is only implemented for MRI)

    Returns:
        List of dataset partitions (one per client)
    """
    if partition_strategy == 'iid':
        partition_size = len(dataset) // num_clients
        lengths = [partition_size] * (num_clients - 1)
        lengths += [len(dataset) - sum(lengths)]
        return random_split(dataset, lengths, torch.Generator().manual_seed(seed))

    if partition_strategy == 'non_iid':
        if dataset_type != 'MRI':
            raise NotImplementedError(f"Non-IID partitioning only implemented for MRI dataset, got {dataset_type}")
        indices = non_iid_mri_indices(dataset.targets, dataset.class_to_idx, num_clients)
        return [Subset(dataset, idx.tolist()) for idx in indices]

    raise ValueError(f"Unknown partition_strategy: {partition_strategy}. Use 'iid' or 'non_iid'")


def load_mri_arrays(data_path="data/MRI", img_size=64):
    """
    Load the Kaggle brain-tumour MRI dataset as grayscale uint8 arrays of shape (N, 1, img_size, img_size).
    The resized arrays are cached in <data_path>/cache_gray_<img_size>.npz after the first call.

    :return: dict with x_train, y_train, x_test, y_test (numpy) and classes
    """
    cache_file = os.path.join(data_path, f"cache_gray_{img_size}.npz")
    if os.path.exists(cache_file):
        with np.load(cache_file) as cached:
            arrays = {key: cached[key] for key in cached.files}
        arrays["classes"] = tuple(arrays["classes"].tolist())
        return arrays

    arrays = {}
    for split, folder_name in (("train", "Training"), ("test", "Testing")):
        folder = datasets.ImageFolder(os.path.join(data_path, folder_name))
        if tuple(folder.classes) != MRI_CLASSES:
            raise ValueError(f"Unexpected MRI classes {folder.classes}, expected {MRI_CLASSES}")
        images = [np.asarray(Image.open(path).convert("L").resize((img_size, img_size), Image.LANCZOS), dtype=np.uint8)
                  for path, _ in folder.samples]
        arrays[f"x_{split}"] = np.stack(images)[:, None]
        arrays[f"y_{split}"] = np.asarray(folder.targets, dtype=np.int64)
    np.savez_compressed(cache_file, classes=np.array(MRI_CLASSES), **arrays)
    arrays["classes"] = MRI_CLASSES
    return arrays


def to_model_input(images_uint8) -> torch.Tensor:
    """uint8 grayscale images -> float tensor in [-1, 1]."""
    return (torch.as_tensor(images_uint8, dtype=torch.float32) / 255.0 - MRI_GRAY_MEAN) / MRI_GRAY_STD


def to_pixels(images) -> torch.Tensor:
    """Model-space images in [-1, 1] -> pixel intensities in [0, 1]."""
    return (images.detach().cpu() * MRI_GRAY_STD + MRI_GRAY_MEAN).clamp(0, 1)


# Define model, architecture and dataset
# The DataLoaders downloads the training and test data that are then normalized.
def load_datasets(num_clients: int, batch_size: int, resize: int, seed: int, num_workers: int, splitter=10,
                  dataset="cifar", data_path="./data/", data_path_val="", use_augmentation=False, 
                  partition_strategy='iid'):
    """
    This function is used to load the dataset and split it into train and test for each client.
    
    Args:
        num_clients: Number of federated clients
        batch_size: Batch size for DataLoaders
        resize: Image resize dimension (e.g., 224 for 224x224)
        seed: Random seed for reproducibility
        num_workers: Number of workers for DataLoader
        splitter: Percentage of data to use for validation (default 10%)
        dataset: Dataset name ('cifar', 'mnist', 'MRI')
        data_path: Path to dataset directory
        data_path_val: Optional separate validation data path
        use_augmentation: If True, applies data augmentation to training data
        partition_strategy: 'iid' (balanced) or 'non_iid' (class-specific) partitioning
    """
    
    # 1. Define base transformations, including normalization
    list_transforms = [transforms.ToTensor(), transforms.Normalize(**NORMALIZE_DICT[dataset])]
    print(dataset)

    # 2. Dataset-specific loading logic
    if dataset == "cifar":
        transformer = transforms.Compose(list_transforms)
        trainset = datasets.CIFAR10(data_path + dataset, train=True, download=True, transform=transformer)
        testset = datasets.CIFAR10(data_path + dataset, train=False, download=True, transform=transformer)
    
    elif dataset == "mnist":
        # MNIST is grayscale (1 channel), so no resize is typically needed, but use the defined transforms
        transformer = transforms.Compose(list_transforms)
        trainset = datasets.MNIST(data_path + dataset, train=True, download=True, transform=transformer)
        testset = datasets.MNIST(data_path + dataset, train=False, download=True, transform=transformer)

    else:
        # Generic ImageFolder logic
        if resize is not None:
            # Build training transforms with optional augmentation
            train_transforms = [transforms.Resize((resize, resize))]
            
            if use_augmentation:
                print("📊 Using data augmentation for training")
                train_transforms.extend([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                ])
            
            train_transforms.extend(list_transforms)
            train_transformer = transforms.Compose(train_transforms)
            
            # Test/val transforms without augmentation
            test_transforms = [transforms.Resize((resize, resize))] + list_transforms
            test_transformer = transforms.Compose(test_transforms)
        else:
            train_transformer = transforms.Compose(list_transforms)
            test_transformer = transforms.Compose(list_transforms)

        trainset = datasets.ImageFolder(data_path + dataset + "/Training", transform=train_transformer)
        testset = datasets.ImageFolder(data_path + dataset + "/Testing", transform=test_transformer)
    
    print(f"The training set is created for the classes : {trainset.classes}")

    # 3. Split training set into `num_clients` partitions (IID or Non-IID)
    print(f"\n🔄 Partitioning strategy: {partition_strategy.upper()}")
    datasets_train = split_data_client(trainset, num_clients, seed, partition_strategy, dataset)
    if data_path_val:
        # Handling external validation set (optional) - no augmentation on validation
        if resize is not None:
            test_transforms = [transforms.Resize((resize, resize))] + [transforms.ToTensor(), transforms.Normalize(**NORMALIZE_DICT[dataset])]
            val_transformer = transforms.Compose(test_transforms)
        else:
            val_transformer = test_transformer
        valset = datasets.ImageFolder(data_path_val, transform=val_transformer)
        datasets_val = split_data_client(valset, num_clients, seed, partition_strategy, dataset)

    # 4. Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for i in range(num_clients):
        if data_path_val:
            # if we already have a validation dataset
            trainloaders.append(DataLoader(datasets_train[i], batch_size=batch_size, shuffle=True, num_workers=num_workers))
            valloaders.append(DataLoader(datasets_val[i], batch_size=batch_size, num_workers=num_workers))
        else:
            # Split training partition into train/val
            len_val = int(len(datasets_train[i]) * splitter / 100) # splitter % validation set
            len_train = len(datasets_train[i]) - len_val
            lengths = [len_train, len_val]
            ds_train, ds_val = random_split(datasets_train[i], lengths, torch.Generator().manual_seed(seed))
            trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers))
            valloaders.append(DataLoader(ds_val, batch_size=batch_size, num_workers=num_workers))

    testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
    return trainloaders, valloaders, testloader