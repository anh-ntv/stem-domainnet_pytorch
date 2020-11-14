import math
import random

random.seed(1)
import numpy as np

np.random.seed(1)

import argparse
from model.digit5 import CNN, Classifier
from model.officecaltech10 import OfficeCaltechNet, OfficeCaltechClassifier
from model.domainnet import DomainNet
from datasets.DigitFive import digit5_dataset_read
from lib.utils.federated_utils import *
from train.train import train, test
from datasets.OfficeCaltech10 import get_office_caltech10_dloader
from datasets.DomainNet_dataset import get_domainnet_dloader
from datasets.DomainNet_dataLoader import get_domainnet_dloader_train
from datasets.Office31 import get_office31_dloader
import os
from os import path
import shutil
import yaml
from scipy.io import loadmat, savemat
import time

# Default settings
parser = argparse.ArgumentParser(description='K3DA Official Implement')
# Dataset Parameters
parser.add_argument("--config", default="DigitFive.yaml")
parser.add_argument('-bp', '--base-path', default=".")
parser.add_argument('-datap', '--data-path', default="../datasets")
parser.add_argument('--target-domain', type=str, help="The target domain we want to perform domain adaptation")
parser.add_argument('--source-domains', type=str, nargs="+", help="The source domains we want to use")
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
# Train Strategy Parameters
parser.add_argument('-t', '--train-time', default=1, type=str,
                    metavar='N', help='the x-th time of training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-dp', '--data-parallel', action='store_false', help='Use Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
# Optimizer Parameters
parser.add_argument('--optimizer', default="SGD", type=str, metavar="Optimizer Name")
parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M', help='Momentum in SGD')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float)
parser.add_argument('-bm', '--bn-momentum', type=float, default=0.1, help="the batchnorm momentum parameter")
parser.add_argument("--gpu", default="0", type=str, metavar='GPU plans to use', help='The GPU id plans to use')
args = parser.parse_args()
# import config files
with open(r"./config/{}".format(args.config)) as file:
    configs = yaml.full_load(file)
# set the visible GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

torch.manual_seed(1)
torch.cuda.manual_seed(1)


def main(args=args, configs=configs):
    src_domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

    num_classes = 345
    train_dloaders = []
    test_dloaders = []

    generator_model = DomainNet(configs["ModelConfig"]["backbone"], args.bn_momentum, configs["ModelConfig"]["pretrained"],
                  args.data_parallel).cuda()

    # create dataLoader for src for testing process (include both train and test data)
    for domain in src_domains:
        source_train_dloader, source_test_dloader = get_domainnet_dloader(args.data_path, domain,
                                                                          configs["TrainingConfig"]["batch_size"],
                                                                          args.workers)
        test_dloaders.append(source_test_dloader)
        train_dloaders.append(source_train_dloader)

    generator_model.eval()

    dataLoader = [train_dloaders, test_dloaders]
    tail_name = ["train", "test"]
    for t_name, d_loaders in zip(tail_name, dataLoader):
        for s_i, domain_name in enumerate(src_domains):
            st = time.time()
            tmp_latent = []
            tmp_label = []
            mat_file= "../datasets/{}_resnet101_{}_pytorch.mat".format(domain_name, t_name)
            print("Processing:", mat_file)
            test_dloader_s = d_loaders[s_i]
            for e in range(10):
                for it, (image_s, label_s) in enumerate(test_dloader_s):
                    image_s = image_s.cuda()
                    with torch.no_grad():
                        output_resnet = generator_model(image_s)
                    tmp_latent.append(output_resnet.cpu().data.numpy())
                    tmp_label.append(label_s.cpu().data.numpy())
            tmp_latent = np.concatenate(tmp_latent)
            tmp_label = np.concatenate(tmp_label)
            print(tmp_latent.shape, tmp_label.shape)
            savemat(mat_file, {'feas': tmp_latent, 'labels': tmp_label})
            print("DONE in", time.time()-st)


def save_checkpoint(state, filename):
    filefolder = "{}/{}/parameter/train_time:{}".format(args.base_path, configs["DataConfig"]["dataset"],
                                                        args.train_time)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename))


if __name__ == "__main__":
    main()
