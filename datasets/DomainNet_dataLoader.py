from os import path
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch


def get_domainnet_data(dataset_path, domain_name, tail_name="train"):
    data_paths = []
    data_labels = []
    # split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    # split_file = path.join("../datasets", "{}_{}.txt".format(domain_name, split))
    split_file = path.join(dataset_path, "{}_{}.txt".format(domain_name, tail_name))
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


class DomainNetLoader_train(Dataset):
    def __init__(self, domain_name_lst, transforms, dataset_path="../datasets"):
        super(DomainNetLoader_train, self).__init__()
        self.transforms = transforms

        self.data_info_lst = []
        min_len = 100000000
        for idx, d_n in enumerate(domain_name_lst):
            # idx must be the same with idx of the classifier for this domain
            data_path_i, data_label_i = get_domainnet_data(dataset_path, d_n, "train")
            if len(data_label_i) < min_len:
                min_len = len(data_label_i)
            self.data_info_lst.append([idx, d_n, data_path_i, data_label_i])
        self.min_data_len = min_len

    def __getitem__(self, index):
        img_lst, label_lst = [], []
        for idx, d_n, data_path_i, data_label_i in self.data_info_lst:
            img = Image.open(data_path_i[index])
            if not img.mode == "RGB":
                img = img.convert("RGB")
            label = data_label_i[index]
            img = self.transforms(img)
            img_lst.append(img)
            label_lst.append(torch.tensor(label))
        return torch.stack(img_lst), torch.stack(label_lst)

    def __len__(self):
        return self.min_data_len


def get_domainnet_dloader_train(dataset_path, domain_name_lst, batch_size, num_workers):
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    # transforms_test = transforms.Compose([
    #     transforms.Resize((224,224)),
    #     transforms.ToTensor()
    # ])
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

    train_dataset = DomainNetLoader_train(domain_name_lst, transforms_train, dataset_path)
    train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                               shuffle=True)
    return train_dloader
