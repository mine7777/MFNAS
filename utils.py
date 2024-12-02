import platform
import os
import re
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset

Dataset2Class = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet-1k-s": 1000,
    "imagenet-1k": 1000,
    "ImageNet16": 1000,
    "ImageNet16-150": 150,
    "ImageNet16-120": 120,
    "ImageNet16-200": 200,
}

def get_datasets(name, root, cutout):

    if name == "cifar10":
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif name == "cifar100":
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif name.startswith("imagenet-1k"):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif name.startswith("ImageNet16"):
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22, 61.26, 65.09]]
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    # Data Argumentation
    if name == "cifar10" or name == "cifar100":
        lists = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if cutout > 0:
            lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        xshape = (1, 3, 32, 32)
    elif name.startswith("ImageNet16"):
        lists = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(16, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if cutout > 0:
            lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        xshape = (1, 3, 16, 16)
    elif name == "tiered":
        lists = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(80, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if cutout > 0:
            lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [
                transforms.CenterCrop(80),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        xshape = (1, 3, 32, 32)
    elif name.startswith("imagenet-1k"):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        if name == "imagenet-1k":
            xlists = [transforms.RandomResizedCrop(224)]
            xlists.append(
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                )
            )
            xlists.append(Lighting(0.1))
        elif name == "imagenet-1k-s":
            xlists = [transforms.RandomResizedCrop(224, scale=(0.2, 1.0))]
        else:
            raise ValueError("invalid name : {:}".format(name))
        xlists.append(transforms.RandomHorizontalFlip(p=0.5))
        xlists.append(transforms.ToTensor())
        xlists.append(normalize)
        train_transform = transforms.Compose(xlists)
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
        xshape = (1, 3, 224, 224)
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    if name == "cifar10":
        train_data = dset.CIFAR10(
            root, train=True, transform=train_transform, download=True
        )
        test_data = dset.CIFAR10(
            root, train=False, transform=test_transform, download=True
        )
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == "cifar100":
        train_data = dset.CIFAR100(
            root, train=True, transform=train_transform, download=True
        )
        test_data = dset.CIFAR100(
            root, train=False, transform=test_transform, download=True
        )
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name.startswith("imagenet-1k"):
        train_data = dset.ImageFolder(osp.join(root, "train"), train_transform)
        test_data = dset.ImageFolder(osp.join(root, "val"), test_transform)
        assert (
            len(train_data) == 1281167 and len(test_data) == 50000
        ), "invalid number of images : {:} & {:} vs {:} & {:}".format(
            len(train_data), len(test_data), 1281167, 50000
        )
    elif name == "ImageNet16":
        train_data = ImageNet16(root, True, train_transform)
        test_data = ImageNet16(root, False, test_transform)
        assert len(train_data) == 1281167 and len(test_data) == 50000
    elif name == "ImageNet16-120":
        train_data = ImageNet16(root, True, train_transform, 120)
        test_data = ImageNet16(root, False, test_transform, 120)
        assert len(train_data) == 151700 and len(test_data) == 6000
    elif name == "ImageNet16-150":
        train_data = ImageNet16(root, True, train_transform, 150)
        test_data = ImageNet16(root, False, test_transform, 150)
        assert len(train_data) == 190272 and len(test_data) == 7500
    elif name == "ImageNet16-200":
        train_data = ImageNet16(root, True, train_transform, 200)
        test_data = ImageNet16(root, False, test_transform, 200)
        assert len(train_data) == 254775 and len(test_data) == 10000
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    class_num = Dataset2Class[name]
    return train_data, test_data, xshape, class_num

def seed_all(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def sum_arr(arr):
    sum = 0.
    for i in range(len(arr)):
        sum += torch.sum(arr[i])
    return sum.item()


def avg_arr(arr):
    sum = 0.
    count = 0.
    for i in range(len(arr)):
        current_count = np.prod([dim for dim in arr[i].shape])
        count += current_count
        sum += torch.sum(arr[i]) * current_count
    return sum.item() / count


def reshape_elements(elements, shapes, device):
    def broadcast_val(elements, shapes):
        ret_grads = []
        for e, sh in zip(elements, shapes):
            ret_grads.append(torch.stack([torch.Tensor(sh).fill_(v) for v in e], dim=0).to(device))
        return ret_grads

    if type(elements[0]) == list:
        outer = []
        for e, sh in zip(elements, shapes):
            outer.append(broadcast_val(e, sh))
        return outer
    else:
        return broadcast_val(elements, shapes)


# Try considering bn too.
def get_layer_metric_array(net, metric, mode):
    metric_array = []

    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))

    return metric_array


def get_default_data_root():
    if 'DATA_PATH' in os.environ:
        return os.environ['DATA_PATH']

    node = platform.node()

    sysname = node
    # To recognize clusters
    """
    if re.match(r'compute-\d+-\d+', sysname):
        sysname = 'cluster1'
    elif re.match(r'node\d', sysname):
        sysname = 'cluster2'
    """

    paths = {
        # Map in the form
        # 'hostname': '/path/to/datasets'
        # Where "datasets" directory is expected to contain CIFAR10, CIFAR100 and ImageNet16 directories
        sysname: '/home/fuwei/data/'
    }

    return paths.get(sysname, None)


def get_nas_archive_path(nasbench_name: str):
    nasbench_name = nasbench_name.lower()
    if nasbench_name not in ('nats_tss', 'nasbench101', 'nats_sss'):
        raise Exception(f"Unknown NAS benchmark {nasbench_name}")

    # Use NATS_PATH or NASBENCH101_PATH env vars to override
    env_var = f'{nasbench_name.upper()}_PATH'
    if env_var in os.environ:
        return os.environ[env_var]

    # Get default path depending on the local system
    node = platform.node()

    sysname = node
    # To recognize clusters
    """
    if re.match(r'compute-\d+-\d+', sysname):
        sysname = 'cluster1'
    elif re.match(r'node\d', sysname):
        sysname = 'cluster2'
    """

    if nasbench_name == 'nats_tss':
        paths = {
            # Map in the form
            # 'hostname': '/path/to/NATS-tss-v1_0-3ffb9-simple'
            sysname: '/home/fuwei/dataset/nasbench/NATS/NATS-tss-v1_0-3ffb9-simple'
        }
    elif nasbench_name == 'nasbench101':
        paths = {
            # Map in the form
            # 'hostname': '/path/to/nb101_pkl_file.pkl'
        }
    elif nasbench_name == 'nats_sss':
        paths = {
            sysname: '/home/fuwei/dataset/nasbench/NATS/NATS-sss-v1_0-50262-simple'
        }
    return paths.get(sysname, None)
