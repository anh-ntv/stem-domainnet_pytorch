import math
import random

random.seed(1)
import numpy as np

np.random.seed(1)

import argparse
from model.digit5 import CNN, Classifier
from model.officecaltech10 import OfficeCaltechNet, OfficeCaltechClassifier
from model.domainnet import DomainNet, DomainNetClassifier, DomainNetDis, DomainNetGAN
from datasets.DigitFive import digit5_dataset_read
from lib.utils.federated_utils import *
from train.train import train_model, test_trg
from datasets.OfficeCaltech10 import get_office_caltech10_dloader
from datasets.DomainNet_dataset import get_domainnet_dloader
from datasets.DomainNet_dataLoader import get_domainnet_dloader_train
from datasets.Office31 import get_office31_dloader
import os
from os import path
import shutil
import yaml

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
parser.add_argument('-mode', '--train-mode', default="1", type=str,
                    metavar='N', help='training mode (default: 1)')
parser.add_argument('-bz', '--batch-size', default=0, type=int,
                    metavar='N', help='batch size to train model (default: 0), it 0, use batch size in default config')
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
import time

torch.manual_seed(1)
torch.cuda.manual_seed(1)


# def main(args=args, configs=configs):
#     src_domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
#     src_domains.remove(args.target_domain)
#
#     num_classes = 345
#     train_dloaders = []
#     test_dloaders = []
#     test_dloaders_train = []
#     classifiers = []
#     optimizer_h_lst = []
#     opt_sche_classifiers = []
#
#     # create dataloader for training process
#     target_train_dloader, target_test_dloader = get_domainnet_dloader(args.data_path,
#                                                                       args.target_domain,
#                                                                       configs["TrainingConfig"]["batch_size"],
#                                                                       args.workers)
#     train_dloaders.append(target_train_dloader)
#     src_train_dloader = get_domainnet_dloader_train(args.data_path, src_domains,
#                                                                       configs["TrainingConfig"]["batch_size"],
#                                                                       args.workers)
#     train_dloaders.append(src_train_dloader)
#
#     # create dataLoader for target for testing process
#     test_dloaders.append(target_test_dloader)
#     test_dloaders_train.append(target_train_dloader)
#
#     generator_model = DomainNet(configs["ModelConfig"]["backbone"], args.bn_momentum, configs["ModelConfig"]["pretrained"],
#                   args.data_parallel).cuda()
#
#     dis_model = DomainNetDis(configs["ModelConfig"]["backbone"], len(src_domains) - 1, args.data_parallel).cuda()
#     gan_model = DomainNetGAN(configs["ModelConfig"]["backbone"], 1, args.data_parallel).cuda()
#
#     classifiers.append(DomainNetClassifier(configs["ModelConfig"]["backbone"], num_classes, args.data_parallel).cuda())
#     args.source_domains = src_domains
#
#     # create dataLoader for src for testing process (include both train and test data)
#     for domain in src_domains:
#         source_train_dloader, source_test_dloader = get_domainnet_dloader(args.data_path, domain,
#                                                                           configs["TrainingConfig"]["batch_size"],
#                                                                           args.workers)
#         test_dloaders.append(source_test_dloader)
#         test_dloaders_train.append(source_train_dloader)
#         classifiers.append(DomainNetClassifier(configs["ModelConfig"]["backbone"], 345, args.data_parallel).cuda())
#
#     optimizer_g = torch.optim.SGD(generator_model.parameters(), momentum=args.momentum,
#                         lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd)
#
#     ##################################
#     optimizer_dis = torch.optim.SGD(dis_model.parameters(), momentum=args.momentum,
#                         lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd)
#     optimizer_gan = torch.optim.SGD(gan_model.parameters(), momentum=args.momentum,
#                         lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd)
#     #################################
#
#     for classifier in classifiers:
#         optimizer_h_lst.append(
#             torch.optim.SGD(classifier.parameters(), momentum=args.momentum,
#                             lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd))
#     # create the optimizer scheduler with cosine annealing schedule
#
#     opt_sche_g = CosineAnnealingLR(optimizer_g, configs["TrainingConfig"]["total_epochs"],
#                           eta_min=configs["TrainingConfig"]["learning_rate_end"])
#     opt_sche_dis = CosineAnnealingLR(optimizer_dis, configs["TrainingConfig"]["total_epochs"],
#                           eta_min=configs["TrainingConfig"]["learning_rate_end"])
#     opt_sche_gan = CosineAnnealingLR(optimizer_gan, configs["TrainingConfig"]["total_epochs"],
#                           eta_min=configs["TrainingConfig"]["learning_rate_end"])
#     for classifier_optimizer in optimizer_h_lst:
#         opt_sche_classifiers.append(
#             CosineAnnealingLR(classifier_optimizer, configs["TrainingConfig"]["total_epochs"],
#                               eta_min=configs["TrainingConfig"]["learning_rate_end"]))
#     # create the event to save log info
#     writer_log_dir = path.join(args.base_path, configs["DataConfig"]["dataset"], "runs",
#                                "train_time:{}".format(args.train_time) + "_" +
#                                args.target_domain + "_" + "_".join(args.source_domains))
#     print("create writer in {}".format(writer_log_dir))
#     if os.path.exists(writer_log_dir):
#         flag = input("{} train_time:{} will be removed, input yes to continue:".format(
#             configs["DataConfig"]["dataset"], args.train_time))
#         if flag == "yes":
#             shutil.rmtree(writer_log_dir, ignore_errors=True)
#     writer = SummaryWriter(log_dir=writer_log_dir)
#     # begin train
#     print("Begin the {} time's training, Dataset:{}, Source Domains {}, Target Domain {}".format(args.train_time,
#                                                                                                  configs[
#                                                                                                      "DataConfig"][
#                                                                                                      "dataset"],
#                                                                                                  args.source_domains,
#                                                                                                  args.target_domain))
#
#     # create the initialized domain weight
#     domain_weight = create_domain_weight(len(args.source_domains))
#     # adjust training strategy with communication round
#     batch_per_epoch, total_epochs = decentralized_training_strategy(
#         communication_rounds=configs["UMDAConfig"]["communication_rounds"],
#         epoch_samples=configs["TrainingConfig"]["epoch_samples"],
#         batch_size=configs["TrainingConfig"]["batch_size"],
#         total_epochs=configs["TrainingConfig"]["total_epochs"])
#     # train generator_model
#     for epoch in range(args.start_epoch, total_epochs):
#         train(train_dloaders, generator_model, dis_model, gan_model, classifiers,
#               optimizer_g, optimizer_dis, optimizer_gan,
#               optimizer_h_lst, epoch, writer, num_classes=num_classes,
#               domain_weight=domain_weight, source_domains=args.source_domains,
#               batch_per_epoch=batch_per_epoch, total_epochs=total_epochs,
#               batchnorm_mmd=configs["UMDAConfig"]["batchnorm_mmd"],
#               communication_rounds=configs["UMDAConfig"]["communication_rounds"],
#               confidence_gate_begin=configs["UMDAConfig"]["confidence_gate_begin"],
#               confidence_gate_end=configs["UMDAConfig"]["confidence_gate_end"],
#               malicious_domain=configs["UMDAConfig"]["malicious"]["attack_domain"],
#               attack_level=configs["UMDAConfig"]["malicious"]["attack_level"])
#         print("TEST_SRC")
#         test(args.target_domain, args.source_domains, test_dloaders, generator_model, classifiers, epoch,
#              writer, num_classes=num_classes, top_5_accuracy=(num_classes > 10), test_iter=200)
#         print("TEST_SRC_TRAIN")
#         test(args.target_domain, args.source_domains, test_dloaders_train, generator_model, classifiers, epoch,
#              writer, num_classes=num_classes, top_5_accuracy=(num_classes > 10), test_iter=200)
#
#         opt_sche_g.step(epoch)
#         opt_sche_dis.step(epoch)
#         opt_sche_gan.step(epoch)
#         for scheduler in opt_sche_classifiers:
#             scheduler.step(epoch)
#         # save generator_model every 10 epochs
#         # if (epoch + 1) % 10 == 0:
#         #     # save target generator_model with epoch, domain, generator_model, optimizer
#         #     save_checkpoint(
#         #         {"epoch": epoch + 1,
#         #          "domain": args.target_domain,
#         #          "backbone": generator_model[0].state_dict(),
#         #          "classifier": classifiers[0].state_dict(),
#         #          "optimizer": optimizer_g[0].state_dict(),
#         #          "classifier_optimizer": optimizer_h_lst[0].state_dict()
#         #          },
#         #         filename="{}.pth.tar".format(args.target_domain))


def main(args=args, configs=configs):
    src_domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    src_domains.remove(args.target_domain)

    num_classes = 345
    train_dloaders = []
    test_dloaders = []
    test_dloaders_train = []
    classifiers = []
    optimizer_h_lst = []
    opt_sche_classifiers = []
    batch_size = args.batch_size
    if batch_size == 0:
        batch_size = configs["TrainingConfig"]["batch_size"]

    # create dataloader for training process
    target_train_dloader, target_test_dloader = get_domainnet_dloader(args.data_path,
                                                                      args.target_domain,
                                                                      batch_size,
                                                                      args.workers)
    train_dloaders.append(target_train_dloader)
    # src_train_dloader = get_domainnet_dloader_train(args.data_path, src_domains,
    #                                                                   batch_size,
    #                                                                   args.workers)
    # train_dloaders.append(src_train_dloader)

    # create dataLoader for target for testing process
    test_dloaders.append(target_test_dloader)
    test_dloaders_train.append(target_train_dloader)

    generator_model = DomainNet(configs["ModelConfig"]["backbone"], args.bn_momentum, configs["ModelConfig"]["pretrained"],
                  args.data_parallel).cuda()

    dis_model = DomainNetDis(configs["ModelConfig"]["backbone"], len(src_domains), args.data_parallel).cuda()
    gan_model = DomainNetGAN(configs["ModelConfig"]["backbone"], 1, args.data_parallel).cuda()

    classifiers.append(DomainNetClassifier(configs["ModelConfig"]["backbone"], num_classes, args.data_parallel).cuda())
    args.source_domains = src_domains

    # create dataLoader for src for testing process (include both train and test data)
    for domain in src_domains:
        source_train_dloader, source_test_dloader = get_domainnet_dloader(args.data_path, domain,
                                                                          batch_size,
                                                                          args.workers)
        test_dloaders.append(source_test_dloader)
        test_dloaders_train.append(source_train_dloader)
        classifiers.append(DomainNetClassifier(configs["ModelConfig"]["backbone"], 345, args.data_parallel).cuda())

    optimizer_g = torch.optim.SGD(generator_model.parameters(), momentum=args.momentum,
                        lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd)
    print("args.data_parallel", args.data_parallel)
    ##################################
    optimizer_dis = torch.optim.SGD(dis_model.parameters(), momentum=args.momentum,
                        lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd)
    optimizer_gan = torch.optim.SGD(gan_model.parameters(), momentum=args.momentum,
                        lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd)
    #################################

    for classifier in classifiers:
        optimizer_h_lst.append(
            torch.optim.SGD(classifier.parameters(), momentum=args.momentum,
                            lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd))
    # create the optimizer scheduler with cosine annealing schedule

    opt_sche_g = CosineAnnealingLR(optimizer_g, configs["TrainingConfig"]["total_epochs"],
                          eta_min=configs["TrainingConfig"]["learning_rate_end"])
    opt_sche_dis = CosineAnnealingLR(optimizer_dis, configs["TrainingConfig"]["total_epochs"],
                          eta_min=configs["TrainingConfig"]["learning_rate_end"])
    opt_sche_gan = CosineAnnealingLR(optimizer_gan, configs["TrainingConfig"]["total_epochs"],
                          eta_min=configs["TrainingConfig"]["learning_rate_end"])
    for classifier_optimizer in optimizer_h_lst:
        opt_sche_classifiers.append(
            CosineAnnealingLR(classifier_optimizer, configs["TrainingConfig"]["total_epochs"],
                              eta_min=configs["TrainingConfig"]["learning_rate_end"]))
    # create the event to save log info
    writer_log_dir = path.join(args.base_path, configs["DataConfig"]["dataset"], "runs",
                               "train_time:{}".format(args.train_time) + "_" +
                               args.target_domain + "_" + "trainMode_{}".format(args.train_mode))
    print("create writer in {}".format(writer_log_dir))
    if os.path.exists(writer_log_dir):
        flag = input("{} train_time:{} will be removed, input yes to continue:".format(
            configs["DataConfig"]["dataset"], args.train_time))
        if flag == "yes":
            shutil.rmtree(writer_log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir=writer_log_dir)
    # begin train
    print("Begin the {} time's training, Dataset:{}, Source Domains {}, Target Domain {}".format(args.train_time,
                                                                                                 configs[
                                                                                                     "DataConfig"][
                                                                                                     "dataset"],
                                                                                                 args.source_domains,
                                                                                                 args.target_domain))

    # create the initialized domain weight
    # domain_weight = create_domain_weight(len(args.source_domains))
    # adjust training strategy with communication round
    batch_per_epoch, total_epochs = decentralized_training_strategy(
        communication_rounds=configs["UMDAConfig"]["communication_rounds"],
        epoch_samples=configs["TrainingConfig"]["epoch_samples"],
        batch_size=batch_size,
        total_epochs=configs["TrainingConfig"]["total_epochs"])
    # train generator_model
    print("batch_per_epoch: {}\t total_epochs: {}".format(batch_per_epoch, total_epochs))
    for epoch in range(args.start_epoch, total_epochs):
        st = time.time()
        print("saving: ", writer_log_dir)
        train_model(train_dloader=test_dloaders_train,
              generator_model=generator_model, classifier_list=classifiers, dis_model=dis_model, gan_model=gan_model,
              optimizer_g=optimizer_g, optimizer_dis=optimizer_dis, optimizer_gan=optimizer_gan,
              classifier_optimizer_list=optimizer_h_lst, epoch=epoch, writer=writer, num_classes=num_classes,
              source_domains=args.source_domains, batchnorm_mmd=configs["UMDAConfig"]["batchnorm_mmd"],
              batch_per_epoch=batch_per_epoch, training_mode=args.train_mode)
        print("TEST")
        test_trg(args.target_domain, args.source_domains, test_dloaders, generator_model, dis_model, gan_model, classifiers, epoch,
             writer, num_classes=num_classes, top_5_accuracy=(num_classes > 10), test_iter=2000, name_summary="Test")
        print("TEST_DATA_TRAIN")
        test_trg(args.target_domain, args.source_domains, test_dloaders_train, generator_model, dis_model, gan_model,
                 classifiers, epoch,
                 writer, num_classes=num_classes, top_5_accuracy=(num_classes > 10), test_iter=200, name_summary="Train")

        opt_sche_g.step(epoch)
        opt_sche_dis.step(epoch)
        opt_sche_gan.step(epoch)
        for scheduler in opt_sche_classifiers:
            scheduler.step(epoch)
        # save generator_model every 10 epochs
        # if (epoch + 1) % 10 == 0:
        #     # save target generator_model with epoch, domain, generator_model, optimizer
        #     save_checkpoint(
        #         {"epoch": epoch + 1,
        #          "domain": args.target_domain,
        #          "backbone": generator_model[0].state_dict(),
        #          "classifier": classifiers[0].state_dict(),
        #          "optimizer": optimizer_g[0].state_dict(),
        #          "classifier_optimizer": optimizer_h_lst[0].state_dict()
        #          },
        #         filename="{}.pth.tar".format(args.target_domain))
        print("time 1 epoch: {:.4f}".format((time.time() - st) / 60))

def save_checkpoint(state, filename):
    filefolder = "{}/{}/parameter/train_time:{}".format(args.base_path, configs["DataConfig"]["dataset"],
                                                        args.train_time)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename))


if __name__ == "__main__":
    main()
