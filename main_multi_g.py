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
from train.train_multi_g import train_model, test_trg
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
parser.add_argument('-te', '--total-epoch', default=0, type=int,
                    metavar='N', help='total_epoch to train model (default: 0), it 0, use total_epoch in default config')
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

torch.manual_seed(1)
torch.cuda.manual_seed(1)


def main(args=args, configs=configs):
    src_domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    src_domains.remove(args.target_domain)

    num_classes = 345
    train_dloaders = []
    test_dloaders = []
    test_dloaders_train = []
    generator_model_lst = []
    classifier_model_lst = []
    optimizer_h_lst = []
    opt_sche_classifiers = []
    batch_size = args.batch_size
    if batch_size == 0:
        batch_size = configs["TrainingConfig"]["batch_size"]

    total_epochs = args.total_epoch
    if total_epochs == 0:
        total_epochs = configs["TrainingConfig"]["total_epochs"]

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
    generator_model_lst.append(DomainNet(configs["ModelConfig"]["backbone"], args.bn_momentum, configs["ModelConfig"]["pretrained"],
                  args.data_parallel).cuda())

    classifier_model_lst.append(DomainNetClassifier(configs["ModelConfig"]["backbone"], num_classes, args.data_parallel).cuda())
    dis_model = DomainNetDis(configs["ModelConfig"]["backbone"], len(src_domains), args.data_parallel).cuda()
    gan_model = DomainNetGAN(configs["ModelConfig"]["backbone"], 1, args.data_parallel).cuda()

    args.source_domains = src_domains

    # create dataLoader for src for testing process (include both train and test data)
    for domain in src_domains:
        source_train_dloader, source_test_dloader = get_domainnet_dloader(args.data_path, domain,
                                                                          batch_size,
                                                                          args.workers)
        test_dloaders.append(source_test_dloader)
        test_dloaders_train.append(source_train_dloader)
        generator_model_lst.append(DomainNet(configs["ModelConfig"]["backbone"], args.bn_momentum, configs["ModelConfig"]["pretrained"],
                  args.data_parallel).cuda())
        classifier_model_lst.append(DomainNetClassifier(configs["ModelConfig"]["backbone"], 345, args.data_parallel).cuda())
    optimizer_g_lst = []
    for g_model in generator_model_lst:
        optimizer_g_lst.append(torch.optim.SGD(g_model.parameters(), momentum=args.momentum,
                        lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd))
        # optimizer_g_lst.append(torch.optim.Adam(g_model.parameters(), lr=0.001
        #                                         , betas=(0.9, 0.999), weight_decay=args.wd))

    ##################################
    optimizer_dis = torch.optim.SGD(dis_model.parameters(), momentum=args.momentum,
                        lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd)
    optimizer_gan = torch.optim.SGD(gan_model.parameters(), momentum=args.momentum,
                        lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd)
    # optimizer_dis = torch.optim.Adam(dis_model.parameters(), lr=0.001
    #                                             , betas=(0.9, 0.999), weight_decay=args.wd)
    # optimizer_gan = torch.optim.Adam(gan_model.parameters(), lr=0.001
    #                                             , betas=(0.9, 0.999), weight_decay=args.wd)
    #################################

    for classifier in classifier_model_lst:
        optimizer_h_lst.append(
            torch.optim.SGD(classifier.parameters(), momentum=args.momentum,
                            lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd))
        # optimizer_h_lst.append(torch.optim.Adam(classifier.parameters(), lr=0.001
        #                      , betas=(0.9, 0.999), weight_decay=args.wd))
    # create the optimizer scheduler with cosine annealing schedule
    opt_sche_g_lst = []
    for opt_g in optimizer_g_lst:
        opt_sche_g_lst.append(CosineAnnealingLR(opt_g, configs["TrainingConfig"]["total_epochs"],
                              eta_min=configs["TrainingConfig"]["learning_rate_end"]))

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
                               "train_multi_g_time_{}".format(args.train_time) + "_" +
                               args.target_domain + "_" + "trainMode_{}".format(args.train_mode))
                               # args.target_domain + "_" + "trainMode_{}_ADAM".format(args.train_mode))
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
        total_epochs=total_epochs)
    # train generator_model
    best_acc = 0
    print("batch_per_epoch: {}\t total_epochs: {}".format(batch_per_epoch, total_epochs))
    conf_gate = 0
    for epoch in range(0, total_epochs):
        print("saving: ", writer_log_dir)
        conf_gate = train_model(train_dloader=test_dloaders_train,
              generator_model=generator_model_lst, classifier_list=classifier_model_lst, dis_model=dis_model, gan_model=gan_model,
              optimizer_g=optimizer_g_lst, optimizer_dis=optimizer_dis, optimizer_gan=optimizer_gan,
              classifier_optimizer_list=optimizer_h_lst, epoch=epoch, writer=writer, num_classes=num_classes,
              source_domains=args.source_domains, batchnorm_mmd=configs["UMDAConfig"]["batchnorm_mmd"],
              batch_per_epoch=batch_per_epoch, training_mode=args.train_mode, confidence_gate=conf_gate)
        best_acc_i = -1
        if epoch % 10 == 0 and (epoch <= 50 or epoch >= 150):
            print("TEST")
            best_acc_i = test_trg(args.target_domain, args.source_domains, test_dloaders, generator_model_lst, dis_model, gan_model, classifier_model_lst, epoch,
                 writer, num_classes=num_classes, top_5_accuracy=(num_classes > 10), test_iter=2000000, name_summary="Test")
            # print("TEST_DATA_TRAIN")
            # test_trg(args.target_domain, args.source_domains, test_dloaders_train, generator_model_lst, dis_model, gan_model,
            #          classifier_model_lst, epoch,
            #          writer, num_classes=num_classes, top_5_accuracy=(num_classes > 10), test_iter=200, name_summary="Train")
        for opt_sche_g in opt_sche_g_lst:
            opt_sche_g.step(epoch)
        opt_sche_dis.step(epoch)
        # opt_sche_gan.step(epoch)
        for scheduler in opt_sche_classifiers:
            scheduler.step(epoch)

        # save target generator_model with epoch, domain, generator_model, optimizer
        if best_acc_i > best_acc:
            best_acc = best_acc_i
            save_lst = {}
            save_lst["epoch"] = epoch
            save_lst["dis"] = dis_model.state_dict()
            save_lst["op_dis"] = optimizer_dis.state_dict()
            idx = 0
            for g, h, opt_g, opt_h in enumerate(generator_model_lst, classifier_model_lst, optimizer_g_lst, optimizer_h_lst):
                g_key = "g_{}".format(idx)
                save_lst[g_key] = g.state_dict()
                h_key = "h_{}".format(idx)
                save_lst[h_key] = h.state_dict()
                op_g_key = "op_g_{}".format(idx)
                save_lst[op_g_key] = opt_g.state_dict()
                op_h_key = "op_h_{}".format(idx)
                save_lst[op_h_key] = opt_h.state_dict()

            save_checkpoint(save_lst, filename="STEM_{}.pth.tar".format(args.target_domain))
    print("TEST")
    test_trg(args.target_domain, args.source_domains, test_dloaders, generator_model_lst, dis_model, gan_model,
             classifier_model_lst, total_epochs,
             writer, num_classes=num_classes, top_5_accuracy=(num_classes > 10), test_iter=2000000,
             name_summary="Test")



def save_checkpoint(state, filename):
    filefolder = "{}/parameter/{}/train_multi_g_time_{}_{}".format(args.base_path, args.target_domain,
                                                        args.train_time, args.train_mode)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    print("saving model at", filefolder)
    torch.save(state, path.join(filefolder, filename))


if __name__ == "__main__":
    main()
