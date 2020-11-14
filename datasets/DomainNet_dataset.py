from os import path
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import random

def read_domainnet_data(dataset_path, domain_name, split="train"):
    data_paths = []
    data_labels = []
    domain_name_lst = domain_name.split(", ")
    # split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    print("reading", domain_name)
    for d_n in domain_name_lst:
        split_file = path.join(dataset_path, "{}_{}.txt".format(d_n, split))
        print(split_file)
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                data_path, label = line.split(' ')
                # data_path = path.join(dataset_path, data_path)
                data_path = path.join("../datasets", data_path)
                label = int(label)
                data_paths.append(data_path)
                data_labels.append(label)
    return data_paths, data_labels

def read_domainnet_data_mergeData(dataset_path, domain_name, split=["train"]):
    data_paths = []
    data_labels = []
    domain_name_lst = domain_name.split(", ")
    # split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    print("reading", domain_name)
    for d_n in domain_name_lst:
        for s_i in split:
            split_file = path.join(dataset_path, "{}_{}.txt".format(d_n, s_i))
            print(split_file)
            with open(split_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    data_path, label = line.split(' ')
                    # data_path = path.join(dataset_path, data_path)
                    data_path = path.join("../datasets", data_path)
                    label = int(label)
                    data_paths.append(data_path)
                    data_labels.append(label)
    return data_paths, data_labels


class DomainNet_dataset(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(DomainNet_dataset, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        random_chose = random.randint(0, len(self.data_paths) - 1)
        img = Image.open(self.data_paths[random_chose])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[random_chose]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)


def get_domainnet_dloader(dataset_path, domain_name, batch_size, num_workers):
    if domain_name == "infograph":
        print("Loading infograph")
        train_data_paths, train_data_labels = read_domainnet_data_mergeData(dataset_path, domain_name, split=["train", "test"])
    else:
        train_data_paths, train_data_labels = read_domainnet_data(dataset_path, domain_name, split="train")
    test_data_paths, test_data_labels = read_domainnet_data(dataset_path, domain_name, split="test")
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1)),
        # transforms.RandomRotation([-60, 60]),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    # transforms_train = transforms.Compose([
    #     transforms.RandomResizedCrop(96, scale=(0.75, 1)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor()
    # ])
    # transforms_test = transforms.Compose([
    #     transforms.Resize((96,96)),
    #     transforms.ToTensor()
    # ])
    # transforms_train = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.ToTensor()
    # ])
    # transforms_test = transforms.Compose([
    #     transforms.ToTensor()
    # ])

    train_dataset = DomainNet_dataset(train_data_paths, train_data_labels, transforms_train, domain_name)
    test_dataset = DomainNet_dataset(test_data_paths, test_data_labels, transforms_test, domain_name)
    train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                               shuffle=True)
    test_dloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                              shuffle=True)
    return train_dloader, test_dloader
