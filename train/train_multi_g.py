import torch
import torch.nn as nn
import numpy as np
from lib.utils.federated_utils import *
from lib.utils.avgmeter import AverageMeter
import torch.nn.functional as F

softmaxcrossEntropy = nn.CrossEntropyLoss().cuda()
softmax_fn = nn.Softmax(dim=1)
sigmoid_fn = nn.Sigmoid()
binaryCrossEntropy = nn.BCELoss().cuda()


# https://github.com/pytorch/pytorch/issues/751
# class StableBCELoss(nn.modules.Module):
class StableBCELoss(torch.nn.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, logit, target):
        logit = logit.view(-1)
        target = target.view(-1)
        neg_abs = - logit.abs()
        loss = logit.clamp(min=0) - logit * target + (1 + neg_abs.exp()).log()
        return loss.mean()


sigmoid_crossEntropy_withLogit_stable = StableBCELoss().cuda()
gan_loss_fn = nn.BCELoss()


# vat code
class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)


cent = ConditionalEntropyLoss().cuda()


class Softmax_entropy(torch.nn.Module):
    def __init__(self):
        super(Softmax_entropy, self).__init__()

    def forward(self, prob, label):
        b = label * prob
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)


softmaxcrossEntropy_fn = Softmax_entropy().cuda()


def loss_phase_1(loss_class_src_lst):
    loss_full = sum(loss_class_src_lst)
    return loss_full


def label_to_1hot(label, num_classes):
    return torch.zeros(label.size(0), num_classes).cuda().scatter_(1, label.view(-1, 1), 1)


def crossEntropy_fn(prob, label, label_to_1hot=False):
    if label_to_1hot:
        num_classes = prob.size(1)
        label = torch.zeros(label.size(0), num_classes).cuda().scatter_(1, label.view(-1, 1), 1)
    return torch.mean(torch.sum(-1.0 * label * torch.log(prob + 1e-10), dim=-1))


def crossEntropy_fn_withConf(prob, label, label_to_1hot=False, confident_thr=0.3):
    if label_to_1hot:
        num_classes = prob.size(1)
        label = torch.zeros(label.size(0), num_classes).cuda().scatter_(1, label.view(-1, 1), 1)
    conf_mask = (torch.max(prob, 1)[0] > confident_thr).float().cuda()
    return torch.mean(conf_mask * torch.sum(-1.0 * label * torch.log(prob + 1e-10), dim=-1))


def softmaxCrossEntropy_withLogit(logit, label):
    return torch.mean(torch.sum(-1.0 * label * torch.log_softmax(logit, dim=1), dim=-1))


def sigmoidCrossEntropy_withLogit(logit, label):
    return binaryCrossEntropy(sigmoid_fn(logit), label)


def compute_acc(out_prob, gt=None, threshold=0.9):
    num_data = out_prob.size(0)
    if gt is None:
        max_v, _ = torch.max(out_prob, dim=-1)
        acc = torch.sum(max_v > threshold) / num_data
    else:

        _, out_prob_top1 = torch.topk(out_prob, k=1, dim=1)
        acc = float(torch.sum(gt.view(-1, 1) == out_prob_top1).item()) / num_data
    return acc


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)  # 900, 900, 2048
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd


def hs_combine_g_x(classifier_list, latent_v, out_dis_model, num_class=None):
    out_hs_sofmax_lst = []
    for classifier in classifier_list[1:]:
        out_hs_sofmax_lst.append(softmax_fn(classifier(latent_v)))

    out_hs_softmax = torch.stack(out_hs_sofmax_lst)  # (num_domain, batch_size, num_class)
    out_hs_softmax = torch.transpose(out_hs_softmax, 0, 1)  # (batch_size, num_domain, num_class)

    out_dis_model_softmax = softmax_fn(out_dis_model)  # (batch_size, num_domain, 1)
    out_dis_model_expand = out_dis_model_softmax.unsqueeze(-1)  # (batch_size, num_domain, 1)
    out_dis_model_repeat = out_dis_model_expand.repeat(
        (1, 1, out_hs_softmax.size(2)))  # (batch_size, num_domain, num_class)

    hs_combine_c_g_x = out_hs_softmax * out_dis_model_repeat  # (batch_size, num_domain, num_class)
    hs_combine_c_g_x = torch.sum(hs_combine_c_g_x, dim=1)  # (batch_size, num_class)

    return hs_combine_c_g_x


def hs_gx_C_gx(out_hs_sofmax_lst, out_dis_model, num_class=None):
    out_hs_softmax = torch.stack(out_hs_sofmax_lst)  # (num_domain, batch_size, num_class)
    out_hs_softmax = torch.transpose(out_hs_softmax, 0, 1)  # (batch_size, num_domain, num_class)

    out_dis_model_softmax = softmax_fn(out_dis_model)  # (batch_size, num_domain, 1)
    out_dis_model_expand = out_dis_model_softmax.unsqueeze(-1)  # (batch_size, num_domain, 1)
    out_dis_model_repeat = out_dis_model_expand.repeat(
        (1, 1, out_hs_softmax.size(2)))  # (batch_size, num_domain, num_class)

    hs_combine_c_g_x = out_hs_softmax * out_dis_model_repeat  # (batch_size, num_domain, num_class)
    hs_combine_c_g_x = torch.sum(hs_combine_c_g_x, dim=1)  # (batch_size, num_class)

    return hs_combine_c_g_x


def train_model(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                num_classes, source_domains, batchnorm_mmd, batch_per_epoch, training_mode=2, confidence_gate=0):
    if training_mode == "3_sep_2":
        train_multi_g_phase_3_separate_2(train_dloader, generator_model, classifier_list, dis_model, gan_model,
                                         optimizer_g,
                                         optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                         num_classes, source_domains, batchnorm_mmd, batch_per_epoch)
    elif training_mode == "4_sep":
        train_multi_g_phase_4_separate(train_dloader, generator_model, classifier_list, dis_model, gan_model,
                                       optimizer_g,
                                       optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                       num_classes, source_domains, batchnorm_mmd, batch_per_epoch)
    elif training_mode == "4_sep_1":
        train_multi_g_phase_4_separate_1(train_dloader, generator_model, classifier_list, dis_model, gan_model,
                                         optimizer_g,
                                         optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                         num_classes, source_domains, batchnorm_mmd, batch_per_epoch)
    elif training_mode == "4_sep_1_1":
        return train_multi_g_phase_4_separate_1_1(train_dloader, generator_model, classifier_list, dis_model, gan_model,
                                           optimizer_g,
                                           optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                           num_classes, source_domains, batchnorm_mmd, batch_per_epoch, confidence_gate)
    elif training_mode == "4_sep_1_1_filter":
        return train_multi_g_phase_4_separate_1_1_filter(train_dloader, generator_model, classifier_list, dis_model, gan_model,
                                                  optimizer_g,
                                                  optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch,
                                                  writer,
                                                  num_classes, source_domains, batchnorm_mmd, batch_per_epoch, confidence_gate)
    elif training_mode == "4_sep_1_1_c":
        train_multi_g_phase_4_separate_1_1_c(train_dloader, generator_model, classifier_list, dis_model, gan_model,
                                             optimizer_g,
                                             optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                             num_classes, source_domains, batchnorm_mmd, batch_per_epoch)
    elif training_mode == "4_sep_1_1_group":
        train_multi_g_phase_4_separate_1_1_group(train_dloader, generator_model, classifier_list, dis_model, gan_model,
                                                 optimizer_g,
                                                 optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                                 num_classes, source_domains, batchnorm_mmd, batch_per_epoch)
    elif training_mode == "4_sep_1_1_mix":
        train_multi_g_phase_4_separate_1_1_mix(train_dloader, generator_model, classifier_list, dis_model, gan_model,
                                               optimizer_g,
                                               optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                               num_classes, source_domains, batchnorm_mmd, batch_per_epoch)
    elif training_mode == "4_sep_1_2":
        train_multi_g_phase_4_separate_1_2(train_dloader, generator_model, classifier_list, dis_model, gan_model,
                                           optimizer_g,
                                           optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                           num_classes, source_domains, batchnorm_mmd, batch_per_epoch)
    elif training_mode == "4_sep_1_2_confi":
        return train_multi_g_phase_4_separate_1_2_confi(train_dloader, generator_model, classifier_list, dis_model, gan_model,
                                           optimizer_g,
                                           optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                           num_classes, source_domains, batchnorm_mmd, batch_per_epoch, confidence_gate)
    elif training_mode == "4_sep_1_2_confi_1":
        return train_multi_g_phase_4_separate_1_2_confi_1(train_dloader, generator_model, classifier_list, dis_model, gan_model,
                                           optimizer_g,
                                           optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                           num_classes, source_domains, batchnorm_mmd, batch_per_epoch, confidence_gate)
    elif training_mode == "4_sep_1_3":
        train_multi_g_phase_4_separate_1_3(train_dloader, generator_model, classifier_list, dis_model, gan_model,
                                           optimizer_g,
                                           optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                           num_classes, source_domains, batchnorm_mmd, batch_per_epoch)
    elif training_mode == "4_sep_1_gan":
        train_multi_g_phase_4_separate_1_gan(train_dloader, generator_model, classifier_list, dis_model, gan_model,
                                             optimizer_g,
                                             optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                             num_classes, source_domains, batchnorm_mmd, batch_per_epoch)
    elif training_mode == "4_sep_2":
        train_multi_g_phase_4_separate_2(train_dloader, generator_model, classifier_list, dis_model, gan_model,
                                         optimizer_g,
                                         optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                         num_classes, source_domains, batchnorm_mmd, batch_per_epoch)
    else:
        print("NO process:", training_mode)


def train_multi_g_phase_4_separate_2(train_dloader, g_model_lst, classifier_list, dis_model, gan_model, optimizer_g_lst,
                                     optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                     num_classes, source_domains, batchnorm_mmd, batch_per_epoch):
    # source_domain_num = len(train_dloader_list[1:])
    print(
        "train_multi_g_phase_4_separate_2, mixup Ht, remove Gan, Gs_i, Gt, C train through hs_combine with PSEUDO LABEL")
    for g_model in g_model_lst:
        g_model.train()
    dis_model.train()
    # gan_model.train()
    for hs_i in classifier_list:
        hs_i.train()
    num_src_domain = 5

    dloader_trg = train_dloader[0]
    loss_class_src_lst = []
    for g_model, opt_g, hs_i, opt_hs_i, dataLoader in zip(g_model_lst[1:], optimizer_g_lst[1:], classifier_list[1:],
                                                          classifier_optimizer_list[1:], train_dloader[1:]):
        for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
            if idx > batch_per_epoch:
                break
            image_s_i = image_s_i.cuda()
            label_s_i = label_s_i.long().cuda()

            # TRAIN PHASE 0
            opt_g.zero_grad()
            opt_hs_i.zero_grad()

            output_hs_i = hs_i(g_model(image_s_i))
            loss_class_src_i = softmaxcrossEntropy(output_hs_i, label_s_i)
            loss_class_src_lst.append(loss_class_src_i)

            loss_class_src_i.backward()
            opt_g.step()
            opt_hs_i.step()

    i = -1
    dloader_src_1 = train_dloader[1]
    dloader_src_2 = train_dloader[2]
    dloader_src_3 = train_dloader[3]
    dloader_src_4 = train_dloader[4]
    dloader_src_5 = train_dloader[5]

    pseudo_l_th = (0.7 - 0.4) * epoch / 60 + 0.4
    mean_conf_dis = AverageMeter()
    for (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        i += 1
        if i == batch_per_epoch:
            break

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]
        optimizer_dis.zero_grad()

        latent_s = []
        label_s = torch.cat(label_s_lst)
        with torch.no_grad():
            for idx in range(len(image_s_lst)):
                latent_s_i = g_model_lst[idx + 1](image_s_lst[idx])
                latent_s.append(latent_s_i)
            latent_s = torch.cat(latent_s)
            out_hs = []
            confident_hs = []
            for classifier in classifier_list[1:]:
                out_hs_i_softmax = softmax_fn(classifier(latent_s))
                out_hs.append(out_hs_i_softmax)
                hs_conf, hs_arg = torch.max(out_hs_i_softmax, dim=-1)

                hs_mask = (hs_arg == label_s).float().cuda()  # if hs_i predict correctly -> hs_mask = 1
                confident_hs_i = hs_mask * hs_conf  # = confident of hs_i when hs_i predict correctly, size: (batch)
                # print("confident_hs_i.size():", confident_hs_i.size())

                confident_hs.append(confident_hs_i)  # size: (num_domain, batch)
            confident_hs = torch.stack(confident_hs)  # size: (num_domain, batch)
            pseudo_label = torch.transpose(confident_hs, 0, 1)  # size: (batch, num_domain)
            ### bo qua truong hop sum(pseudo_label) = 0
            pseudo_l_max, _ = torch.max(pseudo_label, dim=-1)  # size: (batch)
            conf_mask = pseudo_l_max > pseudo_l_th

            # keep the samples have confident > 0.4
            pseudo_label = pseudo_label[conf_mask]
            latent_s = latent_s[conf_mask]
            pseudo_label = pseudo_label / torch.sum(pseudo_label, dim=-1).view(-1,
                                                                               1)  # (batch, num_domain) / (batch, 1)
            # pseudo_label_1hot = label_to_1hot(torch.max(pseudo_label, dim=-1)[1], num_src_domain)

        mean_conf_dis.update(latent_s.size(0))
        out_dis_s = dis_model(latent_s)
        loss_dis_s_pseudo = torch.mean(torch.sum(-1.0 * pseudo_label * torch.log_softmax(out_dis_s, dim=-1), dim=-1))
        loss_dis_s_pseudo.backward()
        optimizer_dis.step()

    mean_confident = AverageMeter()
    for image_t, label_t in dloader_trg:
        image_t = image_t.cuda()
        # label_t = label_t.long().cuda()

        ## TRAIN PHASE 2: Gt + Ht mixup
        optimizer_g_lst[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()

        with torch.no_grad():
            latent_trg = g_model_lst[0](image_t)
            output_dis_trg = dis_model(latent_trg)
            hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)
            hs_g_xt_max_by_class, hs_g_xt_arg_max = hs_g_xt.max(1)
            hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)

        lam = np.random.beta(2, 2)
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        # mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
        mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]
        latent_trg_mixed = g_model_lst[0](mixed_image)
        ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
        # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)

        confidence_gate = (0.95 - 0.7) * (epoch / 60) + 0.7
        if confidence_gate > 0.95:
            confidence_gate = 0.95
        hs_g_xt_mask = (hs_g_xt_max_by_class > confidence_gate).float().cuda()
        mean_confident.update(torch.mean(hs_g_xt_mask))
        loss_mimic_ht_g_xt_mixed = torch.mean(
            hs_g_xt_mask * torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
        # loss_mimic_ht_g_xt_mixed = torch.mean(
        #     torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))

        loss_mimic_ht_g_xt_mixed.backward()
        optimizer_g_lst[0].step()
        classifier_optimizer_list[0].step()
    domain_weight = [1 / (num_src_domain + 1)] * (num_src_domain + 1)
    federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)

    #### MIX Generator\
    print("epoch {}:".format(epoch))
    for n_s, l_s in zip(source_domains, loss_class_src_lst):
        print("\tloss_class_{}: \t{:.4f}".format(n_s, l_s.item()))

    print("\tloss_dis_s_pseudo: {:.4f}".format(loss_dis_s_pseudo.item()))
    print("\tmean_conf_dis: {:.4f}".format(mean_conf_dis.avg))
    print("MIMIC")
    print("\tloss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed.item()))
    print("\tlmean_confident_hs_combine_g_xt: {:.4f}".format(mean_confident.avg))
    # acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
    # acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
    # acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
    # print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
    # print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
    # print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

    ###### print gan_loss_real
    # print("loss_gan_real: {:.4f}".format(loss_gan_real))


def train_multi_g_phase_4_separate_1(train_dloader, g_model_lst, classifier_list, dis_model, gan_model, optimizer_g_lst,
                                     optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                     num_classes, source_domains, batchnorm_mmd, batch_per_epoch):
    # source_domain_num = len(train_dloader_list[1:])
    print("train_multi_g_phase_4_separate_1, mixup Ht, remove Gan, Gs_i, Gt, C and Hs_i train through hs_combine")
    for g_model in g_model_lst:
        g_model.train()
    dis_model.train()
    # gan_model.train()
    for hs_i in classifier_list:
        hs_i.train()
    num_src_domain = 5

    dloader_trg = train_dloader[0]
    loss_class_src_lst = []
    for g_model, opt_g, hs_i, opt_hs_i, dataLoader in zip(g_model_lst[1:], optimizer_g_lst[1:], classifier_list[1:],
                                                          classifier_optimizer_list[1:], train_dloader[1:]):
        for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
            if idx > batch_per_epoch:
                break
            image_s_i = image_s_i.cuda()
            label_s_i = label_s_i.long().cuda()

            # TRAIN PHASE 0
            opt_g.zero_grad()
            opt_hs_i.zero_grad()

            output_hs_i = hs_i(g_model(image_s_i))
            loss_class_src_i = softmaxcrossEntropy(output_hs_i, label_s_i)
            loss_class_src_lst.append(loss_class_src_i)

            loss_class_src_i.backward()
            opt_g.step()
            opt_hs_i.step()

    i = -1
    dloader_src_1 = train_dloader[1]
    dloader_src_2 = train_dloader[2]
    dloader_src_3 = train_dloader[3]
    dloader_src_4 = train_dloader[4]
    dloader_src_5 = train_dloader[5]
    for (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        i += 1
        if i == batch_per_epoch:
            break

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]
        optimizer_dis.zero_grad()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.zero_grad()
        # loss_hs_combine_lst = []
        latent_s = []

        with torch.no_grad():
            for idx in range(len(image_s_lst)):
                latent_s_i = g_model_lst[idx + 1](image_s_lst[idx])
                latent_s.append(latent_s_i)
            latent_s = torch.cat(latent_s)
        out_hs = []
        for classifier in classifier_list[1:]:
            out_hs_i = classifier(latent_s)
            out_hs.append(softmax_fn(out_hs_i))
        out_dis_s = dis_model(latent_s)
        output_hs_combine_g_xs = hs_gx_C_gx(out_hs, out_dis_s)
        label_s = torch.cat(label_s_lst)

        loss_hs_combine = crossEntropy_fn(output_hs_combine_g_xs, label_s, label_to_1hot=True)
        loss_hs_combine.backward()
        optimizer_dis.step()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.step()

    mean_confident = AverageMeter()
    for image_t, label_t in dloader_trg:
        image_t = image_t.cuda()
        # label_t = label_t.long().cuda()

        ## TRAIN PHASE 2: Gt + Ht mixup
        optimizer_g_lst[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()

        with torch.no_grad():
            latent_trg = g_model_lst[0](image_t)
            output_dis_trg = dis_model(latent_trg)
            hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)
            hs_g_xt_max_by_class, hs_g_xt_arg_max = hs_g_xt.max(1)
            hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)

        lam = np.random.beta(2, 2)
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        # mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
        mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]
        latent_trg_mixed = g_model_lst[0](mixed_image)
        ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
        # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)

        confidence_gate = (0.95 - 0.6) * (epoch / 60) + 0.6
        if confidence_gate > 0.95:
            confidence_gate = 0.95
        hs_g_xt_mask = (hs_g_xt_max_by_class > confidence_gate).float().cuda()
        mean_confident.update(torch.mean(hs_g_xt_mask))
        loss_mimic_ht_g_xt_mixed = torch.mean(
            hs_g_xt_mask * torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
        # loss_mimic_ht_g_xt_mixed = torch.mean(
        #     torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))

        loss_mimic_ht_g_xt_mixed.backward()
        optimizer_g_lst[0].step()
        classifier_optimizer_list[0].step()
    domain_weight = [1 / (num_src_domain + 1)] * (num_src_domain + 1)
    federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)

    #### MIX Generator\
    print("epoch {}:".format(epoch))
    for n_s, l_s in zip(source_domains, loss_class_src_lst):
        print("\tloss_class_{}: \t{:.4f}".format(n_s, l_s.item()))

    print("\tloss_hs_combine_g_xs: {:.4f}".format(loss_hs_combine.item()))
    print("MIMIC")
    print("\tloss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed.item()))
    print("\tlmean_confident_hs_combine_g_xt: {:.4f}".format(mean_confident.avg))
    # acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
    # acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
    # acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
    # print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
    # print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
    # print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

    ###### print gan_loss_real
    # print("loss_gan_real: {:.4f}".format(loss_gan_real))


def train_multi_g_phase_4_separate_1_1(train_dloader, g_model_lst, classifier_list, dis_model, gan_model,
                                       optimizer_g_lst,
                                       optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                       num_classes, source_domains, batchnorm_mmd, batch_per_epoch, confidence_gate):
    # source_domain_num = len(train_dloader_list[1:])
    print(
        "train_multi_g_phase_4_separate_1_1, mixup Ht, remove Gan, Gs_i, Gt, C and Hs_i train through hs_combine, update G_combine by Gs only, before train gt")
    for g_model in g_model_lst:
        g_model.train()
    dis_model.train()
    # gan_model.train()
    for hs_i in classifier_list:
        hs_i.train()
    num_src_domain = len(source_domains)

    dloader_trg = train_dloader[0]
    loss_class_src_lst = []
    for g_model, opt_g, hs_i, opt_hs_i, dataLoader in zip(g_model_lst[1:], optimizer_g_lst[1:], classifier_list[1:],
                                                          classifier_optimizer_list[1:], train_dloader[1:]):
        for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
            if idx > batch_per_epoch:
                break
            image_s_i = image_s_i.cuda()
            label_s_i = label_s_i.long().cuda()

            # TRAIN PHASE 0
            opt_g.zero_grad()
            opt_hs_i.zero_grad()

            output_hs_i = hs_i(g_model(image_s_i))
            loss_class_src_i = softmaxcrossEntropy(output_hs_i, label_s_i)
            loss_class_src_lst.append(loss_class_src_i)

            loss_class_src_i.backward()
            opt_g.step()
            opt_hs_i.step()

    domain_weight = [1 / num_src_domain] * num_src_domain
    domain_weight.insert(0, 0)
    federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)

    i = -1
    dloader_src_1 = train_dloader[1]
    dloader_src_2 = train_dloader[2]
    dloader_src_3 = train_dloader[3]
    dloader_src_4 = train_dloader[4]
    dloader_src_5 = train_dloader[5]
    for (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
    # for (image_s_1, label_s_1), (image_s_2, label_s_2) \
    #         in zip(dloader_src_1, dloader_src_2):
        i += 1
        if i == batch_per_epoch:
            break

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]

        # image_s_lst = [
        #     image_s_1.cuda(), image_s_2.cuda()
        # ]
        # label_s_lst = [
        #     label_s_1.long().cuda(), label_s_2.long().cuda()
        # ]
        optimizer_dis.zero_grad()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.zero_grad()
        # loss_hs_combine_lst = []
        latent_s = []

        with torch.no_grad():
            for idx in range(len(image_s_lst)):
                latent_s_i = g_model_lst[idx + 1](image_s_lst[idx])
                latent_s.append(latent_s_i)
            latent_s = torch.cat(latent_s)
        out_hs = []
        for classifier in classifier_list[1:]:
            out_hs_i = classifier(latent_s)
            out_hs.append(softmax_fn(out_hs_i))
        out_dis_s = dis_model(latent_s)
        output_hs_combine_g_xs = hs_gx_C_gx(out_hs, out_dis_s)
        label_s = torch.cat(label_s_lst)

        loss_hs_combine = crossEntropy_fn(output_hs_combine_g_xs, label_s, label_to_1hot=True)
        loss_hs_combine.backward()
        optimizer_dis.step()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.step()

    # confidence_gate = (0.95 - 0.6) * (epoch / 60) + 0.6
    # if confidence_gate > 0.95:
    #     confidence_gate = 0.95

    mean_confident_gate = AverageMeter()
    mean_confident = AverageMeter()
    for image_t, label_t in dloader_trg:
        image_t = image_t.cuda()
        # label_t = label_t.long().cuda()

        ## TRAIN PHASE 2: Gt + Ht mixup
        # optimizer_g_lst[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()

        with torch.no_grad():
            latent_trg = g_model_lst[0](image_t)
            output_dis_trg = dis_model(latent_trg)
            hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)
            hs_g_xt_max_by_class, hs_g_xt_arg_max = hs_g_xt.max(1)
            hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)

        lam = np.random.beta(2, 2)
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        # mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
        mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]
        with torch.no_grad():
            latent_trg_mixed = g_model_lst[0](mixed_image)
        ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
        hs_g_xt_conf, hs_g_xt_arg_max = hs_g_xt.max(1)
        # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)
        confidence_gate_i = torch.mean(hs_g_xt_conf.float()) * 1.2
        if epoch == 0:
            confidence_gate = confidence_gate_i.item()
        mean_confident_gate.update(confidence_gate_i.item())
        if confidence_gate < 0.3:
            confidence_gate = 0.3
        # print("mean conf", torch.mean(hs_g_xt_conf.float()).item())
        hs_g_xt_mask = (hs_g_xt_conf > confidence_gate).float().cuda()
        # hs_g_xt_mask = (hs_g_xt_max_by_class > confidence_gate).float().cuda()

        mean_confident.update(torch.mean(hs_g_xt_mask))
        loss_mimic_ht_g_xt_mixed = torch.mean(
            hs_g_xt_mask * torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
        # loss_mimic_ht_g_xt_mixed = torch.mean(
        #     torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))

        loss_mimic_ht_g_xt_mixed.backward()
        # optimizer_g_lst[0].step()
        classifier_optimizer_list[0].step()

    #### MIX Generator\
    print("epoch {}:".format(epoch))
    for n_s, l_s in zip(source_domains, loss_class_src_lst):
        print("\tloss_class_{}: \t{:.4f}".format(n_s, l_s.item()))

    print("\tloss_hs_combine_g_xs: {:.4f}".format(loss_hs_combine.item()))
    print("MIMIC")
    print("\tloss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed.item()))
    print("\tmean_confident_hs_combine_g_xt: {:.4f}".format(mean_confident.avg))
    print("\tconfidence_gate: {:.4f}".format(confidence_gate))
    # acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
    # acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
    # acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
    # print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
    # print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
    # print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

    ###### print gan_loss_real
    # print("loss_gan_real: {:.4f}".format(loss_gan_real))

    if epoch == 0:
        confidence_gate = mean_confident_gate.avg
    else:
        confidence_gate = 0.8 * confidence_gate + 0.2 * mean_confident_gate.avg
    return confidence_gate


def train_multi_g_phase_4_separate_1_1_filter(train_dloader, g_model_lst, classifier_list, dis_model, gan_model,
                                              optimizer_g_lst,
                                              optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                              num_classes, source_domains, batchnorm_mmd, batch_per_epoch, confidence_gate):
    # print(
    #     "train_multi_g_phase_4_separate_1_1_filter_7, mixup Ht, remove Gan, Gs_i, Gt, C and Hs_i train through hs_combine, update G_combine by Gs only, before train gt")
    # for g_model in g_model_lst:
    #     g_model.train()
    # dis_model.train()
    # # gan_model.train()
    # for hs_i in classifier_list:
    #     hs_i.train()
    # num_src_domain = len(source_domains)
    #
    # loss_class_src_lst = []
    # for g_model, opt_g, hs_i, opt_hs_i, dataLoader in zip(g_model_lst[1:], optimizer_g_lst[1:], classifier_list[1:],
    #                                                       classifier_optimizer_list[1:], train_dloader[1:]):
    #     for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
    #         if idx > batch_per_epoch:
    #             break
    #         image_s_i = image_s_i.cuda()
    #         label_s_i = label_s_i.long().cuda()
    #
    #         # TRAIN PHASE 0
    #         opt_g.zero_grad()
    #         opt_hs_i.zero_grad()
    #
    #         output_hs_i = hs_i(g_model(image_s_i))
    #         loss_class_src_i = softmaxcrossEntropy(output_hs_i, label_s_i)
    #         loss_class_src_lst.append(loss_class_src_i)
    #
    #         loss_class_src_i.backward()
    #         opt_g.step()
    #         opt_hs_i.step()
    #
    #
    # domain_weight = [1 / num_src_domain] * num_src_domain
    # domain_weight.insert(0, 0)
    # # domain_weight = [1 / (num_src_domain + 1)] * (num_src_domain + 1)
    # federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd, update_all=False)
    #
    # loss_mimic_ht_g_xt_mixed = 0
    # mean_confident = AverageMeter()
    # mean_confident_gate = AverageMeter()
    #
    # i = -1
    # dloader_trg = train_dloader[0]
    # dloader_src_1 = train_dloader[1]
    # dloader_src_2 = train_dloader[2]
    # dloader_src_3 = train_dloader[3]
    # dloader_src_4 = train_dloader[4]
    # dloader_src_5 = train_dloader[5]
    #
    # for (image_t, label_t), (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
    #         image_s_4, label_s_4), (image_s_5, label_s_5) \
    #         in zip(dloader_trg, dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
    #     i += 1
    #     if i > batch_per_epoch:
    #         break
    #
    #     image_s_lst = [
    #         image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
    #     ]
    #     label_s_lst = [
    #         label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
    #         label_s_5.long().cuda()
    #     ]
    #
    #     optimizer_dis.zero_grad()
    #     for opt_hs_i in classifier_optimizer_list[1:]:
    #         opt_hs_i.zero_grad()
    #     # loss_hs_combine_lst = []
    #     latent_s = []
    #
    #     with torch.no_grad():
    #         for idx in range(len(image_s_lst)):
    #             latent_s_i = g_model_lst[0](image_s_lst[idx])
    #             latent_s.append(latent_s_i)
    #         latent_s = torch.cat(latent_s)
    #     out_hs = []
    #     for classifier in classifier_list[1:]:
    #         out_hs_i = classifier(latent_s)
    #         out_hs.append(softmax_fn(out_hs_i))
    #     out_dis_s = dis_model(latent_s)
    #     output_hs_combine_g_xs = hs_gx_C_gx(out_hs, out_dis_s)
    #     label_s = torch.cat(label_s_lst)
    #
    #     loss_hs_combine = crossEntropy_fn(output_hs_combine_g_xs, label_s, label_to_1hot=True)
    #     loss_hs_combine.backward()
    #     # for opt_g in optimizer_g_lst[1:]:
    #     #     opt_g.step()
    #     # optimizer_g_lst[0].step()
    #     optimizer_dis.step()
    #     for opt_hs_i in classifier_optimizer_list[1:]:
    #         opt_hs_i.step()
    #
    # # loss_mimic_ht_g_xt_mixed = 0
    # # mean_confident = AverageMeter()
    # # mean_confident_gate = AverageMeter()
    # # i = -1
    # # for image_t, label_t in dloader_trg:
    #     image_t = image_t.cuda()
    #     # i += 1
    #     # if i > batch_per_epoch:
    #     #     break
    #     # label_t = label_t.long().cuda()
    #
    #     ## TRAIN PHASE 2: Gt + Ht mixup
    #     optimizer_g_lst[0].zero_grad()
    #     classifier_optimizer_list[0].zero_grad()
    #
    #     latent_trg = g_model_lst[0](image_t)
    #     with torch.no_grad():
    #         output_dis_trg = dis_model(latent_trg)
    #         hs_g_xt_softmax_lst = []
    #         for g_s, h_s in zip(g_model_lst[1:], classifier_list[1:]):
    #             hs_g_xt_softmax_lst.append(
    #                 softmax_fn(h_s(g_s(image_t)))
    #             )
    #         hs_g_xt = hs_gx_C_gx(hs_g_xt_softmax_lst, output_dis_trg)
    #
    #         hs_g_xt_conf, hs_g_xt_arg_max = hs_g_xt.max(1)
    #         hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)
    #         confidence_gate_i = torch.mean(hs_g_xt_conf.float()) * 1.2
    #         if epoch == 0:
    #             confidence_gate = confidence_gate_i.item()
    #         mean_confident_gate.update(confidence_gate_i.item())
    #         if confidence_gate < 0.3:
    #             confidence_gate = 0.3
    #         # print("mean conf", torch.mean(hs_g_xt_conf.float()).item())
    #         hs_g_xt_mask = hs_g_xt_conf > confidence_gate
    #         # if i % 500 == 0:
    #         #     print("confidence_gate", confidence_gate)
    #         #     print("torch.mean(hs_g_xt_mask.float())", torch.mean(hs_g_xt_mask.float()))
    #         image_t = image_t[hs_g_xt_mask]
    #         hs_g_xt_onehot = hs_g_xt_onehot[hs_g_xt_mask]
    #
    #         if image_t.size(0) < 1.0:
    #             continue
    #         lam = np.random.beta(2, 2)
    #         num_train_data = image_t.size(0)
    #         index = torch.randperm(num_train_data).cuda()
    #         mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
    #         # mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
    #         mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]
    #         # with torch.no_grad():
    #     latent_trg_mixed = g_model_lst[0](mixed_image)
    #     ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
    #     # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)
    #
    #     mean_confident.update(torch.mean(hs_g_xt_mask.float()))
    #     loss_mimic_ht_g_xt_mixed = torch.mean(
    #         torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
    #     # loss_mimic_ht_g_xt_mixed = torch.mean(
    #     #     torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
    #
    #     loss_mimic_ht_g_xt_mixed.backward()
    #     optimizer_g_lst[0].step()
    #     classifier_optimizer_list[0].step()
    #
    #
    # # domain_weight = [1 / num_src_domain] * num_src_domain
    # # domain_weight.insert(0, 0)
    # domain_weight = [1 / (num_src_domain + 1)] * (num_src_domain + 1)
    # federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)
    #
    # #### MIX Generator\
    # print("epoch {}:".format(epoch))
    # for n_s, l_s in zip(source_domains, loss_class_src_lst):
    #     print("\tloss_class_{}: \t{:.4f}".format(n_s, l_s.item()))
    #
    # print("\tloss_hs_combine_g_xs: {:.4f}".format(loss_hs_combine.item()))
    # print("MIMIC")
    # print("\tloss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed))
    # print("\tmean_confident_hs_combine_g_xt: {:.4f}".format(mean_confident.avg))
    # print("\tconfidence_gate: {:.4f}".format(confidence_gate))
    #
    # if epoch == 0:
    #     confidence_gate = mean_confident_gate.avg
    # else:
    #     confidence_gate = 0.8 * confidence_gate + 0.2 * mean_confident_gate.avg
    # return confidence_gate

    print(
        "train_multi_g_phase_4_separate_1_1_filter_6, mixup Ht, remove Gan, Gs_i, Gt, C and Hs_i train through hs_combine, update G_combine by Gs only, before train gt")
    for g_model in g_model_lst:
        g_model.train()
    dis_model.train()
    # gan_model.train()
    for hs_i in classifier_list:
        hs_i.train()
    num_src_domain = len(source_domains)

    loss_class_src_lst = []
    for g_model, opt_g, hs_i, opt_hs_i, dataLoader in zip(g_model_lst[1:], optimizer_g_lst[1:], classifier_list[1:],
                                                          classifier_optimizer_list[1:], train_dloader[1:]):
        for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
            if idx > batch_per_epoch:
                break
            image_s_i = image_s_i.cuda()
            label_s_i = label_s_i.long().cuda()

            # TRAIN PHASE 0
            opt_g.zero_grad()
            opt_hs_i.zero_grad()

            output_hs_i = hs_i(g_model(image_s_i))
            loss_class_src_i = softmaxcrossEntropy(output_hs_i, label_s_i)
            loss_class_src_lst.append(loss_class_src_i)

            loss_class_src_i.backward()
            opt_g.step()
            opt_hs_i.step()

    i = -1
    dloader_trg = train_dloader[0]
    dloader_src_1 = train_dloader[1]
    dloader_src_2 = train_dloader[2]
    dloader_src_3 = train_dloader[3]
    dloader_src_4 = train_dloader[4]
    dloader_src_5 = train_dloader[5]
    for (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        i += 1
        if i > batch_per_epoch:
            break

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]

        optimizer_dis.zero_grad()
        # for opt_hs_i in classifier_optimizer_list[1:]:
        #     opt_hs_i.zero_grad()
        # loss_hs_combine_lst = []
        latent_s = []

        with torch.no_grad():
            for idx in range(len(image_s_lst)):
                latent_s_i = g_model_lst[idx + 1](image_s_lst[idx])
                latent_s.append(latent_s_i)
            latent_s = torch.cat(latent_s)
            out_hs = []
            for classifier in classifier_list[1:]:
                out_hs_i = classifier(latent_s)
                out_hs.append(softmax_fn(out_hs_i))
        out_dis_s = dis_model(latent_s)
        output_hs_combine_g_xs = hs_gx_C_gx(out_hs, out_dis_s)
        label_s = torch.cat(label_s_lst)

        loss_hs_combine = crossEntropy_fn(output_hs_combine_g_xs, label_s, label_to_1hot=True)
        loss_hs_combine.backward()
        # for opt_g in optimizer_g_lst[1:]:
        #     opt_g.step()
        # optimizer_g_lst[0].step()
        optimizer_dis.step()
        # for opt_hs_i in classifier_optimizer_list[1:]:
        #     opt_hs_i.step()

    domain_weight = [1 / num_src_domain] * num_src_domain
    domain_weight.insert(0, 0)
    # domain_weight = [1 / (num_src_domain + 1)] * (num_src_domain + 1)
    federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd, update_all=False)

    loss_mimic_ht_g_xt_mixed = 0
    mean_confident = AverageMeter()
    mean_confident_gate = AverageMeter()
    i = -1
    for image_t, label_t in dloader_trg:
        image_t = image_t.cuda()
        i += 1
        if i > batch_per_epoch:
            break
        # label_t = label_t.long().cuda()

        ## TRAIN PHASE 2: Gt + Ht mixup
        optimizer_g_lst[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()

        latent_trg = g_model_lst[0](image_t)
        with torch.no_grad():
            output_dis_trg = dis_model(latent_trg)
            hs_g_xt_softmax_lst = []
            for g_s, h_s in zip(g_model_lst[1:], classifier_list[1:]):
                hs_g_xt_softmax_lst.append(
                    softmax_fn(h_s(g_s(image_t)))
                )
            hs_g_xt = hs_gx_C_gx(hs_g_xt_softmax_lst, output_dis_trg)

            hs_g_xt_conf, hs_g_xt_arg_max = hs_g_xt.max(1)
            hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)
            confidence_gate_i = torch.mean(hs_g_xt_conf.float()) * 1.2
            if epoch == 0:
                confidence_gate = confidence_gate_i.item()
            mean_confident_gate.update(confidence_gate_i.item())
            if confidence_gate < 0.3:
                confidence_gate = 0.3
            # print("mean conf", torch.mean(hs_g_xt_conf.float()).item())
            hs_g_xt_mask = hs_g_xt_conf > confidence_gate
            # if i % 500 == 0:
            #     print("confidence_gate", confidence_gate)
            #     print("torch.mean(hs_g_xt_mask.float())", torch.mean(hs_g_xt_mask.float()))
            image_t = image_t[hs_g_xt_mask]
            hs_g_xt_onehot = hs_g_xt_onehot[hs_g_xt_mask]

            if image_t.size(0) < 1.0:
                continue
            lam = np.random.beta(2, 2)
            num_train_data = image_t.size(0)
            index = torch.randperm(num_train_data).cuda()
            mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
            # mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
            mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]
            # with torch.no_grad():
        latent_trg_mixed = g_model_lst[0](mixed_image)
        ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
        # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)

        mean_confident.update(torch.mean(hs_g_xt_mask.float()))
        loss_mimic_ht_g_xt_mixed = torch.mean(
            torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
        # loss_mimic_ht_g_xt_mixed = torch.mean(
        #     torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))

        loss_mimic_ht_g_xt_mixed.backward()
        optimizer_g_lst[0].step()
        classifier_optimizer_list[0].step()


    # domain_weight = [1 / num_src_domain] * num_src_domain
    # domain_weight.insert(0, 0)
    domain_weight = [1 / (num_src_domain + 1)] * (num_src_domain + 1)
    federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)

    #### MIX Generator\
    print("epoch {}:".format(epoch))
    for n_s, l_s in zip(source_domains, loss_class_src_lst):
        print("\tloss_class_{}: \t{:.4f}".format(n_s, l_s.item()))

    print("\tloss_hs_combine_g_xs: {:.4f}".format(loss_hs_combine.item()))
    print("MIMIC")
    print("\tloss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed))
    print("\tmean_confident_hs_combine_g_xt: {:.4f}".format(mean_confident.avg))
    print("\tconfidence_gate: {:.4f}".format(confidence_gate))

    if epoch == 0:
        confidence_gate = mean_confident_gate.avg
    else:
        confidence_gate = 0.8 * confidence_gate + 0.2 * mean_confident_gate.avg
    return confidence_gate

#   print(
#         "train_multi_g_phase_4_separate_1_1_filter_5, mixup Ht, remove Gan, Gs_i, Gt, C and Hs_i train through hs_combine, update G_combine by Gs only, before train gt")
#     for g_model in g_model_lst:
#         g_model.train()
#     dis_model.train()
#     # gan_model.train()
#     for hs_i in classifier_list:
#         hs_i.train()
#     num_src_domain = len(source_domains)
#
#     loss_class_src_lst = []
#     for g_model, opt_g, hs_i, opt_hs_i, dataLoader in zip(g_model_lst[1:], optimizer_g_lst[1:], classifier_list[1:],
#                                                           classifier_optimizer_list[1:], train_dloader[1:]):
#         for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
#             if idx > batch_per_epoch:
#                 break
#             image_s_i = image_s_i.cuda()
#             label_s_i = label_s_i.long().cuda()
#
#             # TRAIN PHASE 0
#             opt_g.zero_grad()
#             opt_hs_i.zero_grad()
#
#             output_hs_i = hs_i(g_model(image_s_i))
#             loss_class_src_i = softmaxcrossEntropy(output_hs_i, label_s_i)
#             loss_class_src_lst.append(loss_class_src_i)
#
#             loss_class_src_i.backward()
#             opt_g.step()
#             opt_hs_i.step()
#
#
#     domain_weight = [1 / num_src_domain] * num_src_domain
#     domain_weight.insert(0, 0)
#     # domain_weight = [1 / (num_src_domain + 1)] * (num_src_domain + 1)
#     federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)
#
#     i = -1
#     dloader_trg = train_dloader[0]
#     dloader_src_1 = train_dloader[1]
#     dloader_src_2 = train_dloader[2]
#     dloader_src_3 = train_dloader[3]
#     dloader_src_4 = train_dloader[4]
#     dloader_src_5 = train_dloader[5]
#     for (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
#             image_s_4, label_s_4), (image_s_5, label_s_5) \
#             in zip(dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
#         i += 1
#         if i > batch_per_epoch:
#             break
#
#         image_s_lst = [
#             image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
#         ]
#         label_s_lst = [
#             label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
#             label_s_5.long().cuda()
#         ]
#
#         optimizer_g_lst[0].zero_grad()
#         optimizer_dis.zero_grad()
#         for opt_hs_i in classifier_optimizer_list[1:]:
#             opt_hs_i.zero_grad()
#         # loss_hs_combine_lst = []
#         latent_s = []
#
#         # with torch.no_grad():
#         for idx in range(len(image_s_lst)):
#             latent_s_i = g_model_lst[0](image_s_lst[idx])
#             latent_s.append(latent_s_i)
#         latent_s = torch.cat(latent_s)
#         out_hs = []
#         for classifier in classifier_list[1:]:
#             out_hs_i = classifier(latent_s)
#             out_hs.append(softmax_fn(out_hs_i))
#         out_dis_s = dis_model(latent_s)
#         output_hs_combine_g_xs = hs_gx_C_gx(out_hs, out_dis_s)
#         label_s = torch.cat(label_s_lst)
#
#         loss_hs_combine = crossEntropy_fn(output_hs_combine_g_xs, label_s, label_to_1hot=True)
#         loss_hs_combine.backward()
#         # for opt_g in optimizer_g_lst[1:]:
#         #     opt_g.step()
#         optimizer_g_lst[0].step()
#         optimizer_dis.step()
#         for opt_hs_i in classifier_optimizer_list[1:]:
#             opt_hs_i.step()
#
#     loss_mimic_ht_g_xt_mixed = 0
#     mean_confident = AverageMeter()
#     mean_confident_gate = AverageMeter()
#     i = -1
#     for image_t, label_t in dloader_trg:
#         image_t = image_t.cuda()
#         i += 1
#         if i > batch_per_epoch:
#             break
#         # label_t = label_t.long().cuda()
#
#         ## TRAIN PHASE 2: Gt + Ht mixup
#         # optimizer_g_lst[0].zero_grad()
#         classifier_optimizer_list[0].zero_grad()
#
#         with torch.no_grad():
#             latent_trg = g_model_lst[0](image_t)
#             output_dis_trg = dis_model(latent_trg)
#             hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)
#             hs_g_xt_conf, hs_g_xt_arg_max = hs_g_xt.max(1)
#             hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)
#             confidence_gate_i = torch.mean(hs_g_xt_conf.float()) * 1.2
#             if epoch == 0:
#                 confidence_gate = confidence_gate_i.item()
#             mean_confident_gate.update(confidence_gate_i.item())
#             # if confidence_gate < 0.3:
#             #     confidence_gate = 0.3
#             # print("mean conf", torch.mean(hs_g_xt_conf.float()).item())
#             hs_g_xt_mask = hs_g_xt_conf > confidence_gate
#             # if i % 500 == 0:
#             #     print("confidence_gate", confidence_gate)
#             #     print("torch.mean(hs_g_xt_mask.float())", torch.mean(hs_g_xt_mask.float()))
#             image_t = image_t[hs_g_xt_mask, :]
#             hs_g_xt_onehot = hs_g_xt_onehot[hs_g_xt_mask]
#
#             if image_t.size(0) < 1.0:
#                 continue
#             lam = np.random.beta(2, 2)
#             num_train_data = image_t.size(0)
#             index = torch.randperm(num_train_data).cuda()
#             mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
#             # mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
#             mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]
#             # with torch.no_grad():
#             latent_trg_mixed = g_model_lst[0](mixed_image)
#         ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
#         # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)
#
#         mean_confident.update(torch.mean(hs_g_xt_mask.float()))
#         loss_mimic_ht_g_xt_mixed = torch.mean(
#             torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
#         # loss_mimic_ht_g_xt_mixed = torch.mean(
#         #     torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
#
#         loss_mimic_ht_g_xt_mixed.backward()
#         # optimizer_g_lst[0].step()
#         classifier_optimizer_list[0].step()
#
#     if epoch == 0:
#         confidence_gate = mean_confident_gate.avg
#     else:
#         confidence_gate = 0.8 * confidence_gate + 0.2 * mean_confident_gate.avg
#
#     # domain_weight = [1 / num_src_domain] * num_src_domain
#     # domain_weight.insert(0, 0)
#     domain_weight = [1 / (num_src_domain + 1)] * (num_src_domain + 1)
#     federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)
#
#     #### MIX Generator\
#     print("epoch {}:".format(epoch))
#     for n_s, l_s in zip(source_domains, loss_class_src_lst):
#         print("\tloss_class_{}: \t{:.4f}".format(n_s, l_s.item()))
#
#     print("\tloss_hs_combine_g_xs: {:.4f}".format(loss_hs_combine.item()))
#     print("MIMIC")
#     print("\tloss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed))
#     print("\tmean_confident_hs_combine_g_xt: {:.4f}".format(mean_confident.avg))
#     print("\tconfidence_gate: {:.4f}".format(confidence_gate))
#     return confidence_gate

    # print(
    #     "train_multi_g_phase_4_separate_1_1_filter_4, mixup Ht, remove Gan, Gs_i, Gt, C and Hs_i train through hs_combine, update G_combine by Gs only, before train gt")
    # for g_model in g_model_lst:
    #     g_model.train()
    # dis_model.train()
    # # gan_model.train()
    # for hs_i in classifier_list:
    #     hs_i.train()
    # num_src_domain = len(source_domains)
    #
    # # domain_weight = [1 / num_src_domain] * num_src_domain
    # # domain_weight.insert(0, 0)
    # domain_weight = [1 / (num_src_domain + 1)] * (num_src_domain + 1)
    # federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)
    #
    # loss_class_src_lst = []
    # for g_model, opt_g, hs_i, opt_hs_i, dataLoader in zip(g_model_lst[1:], optimizer_g_lst[1:], classifier_list[1:],
    #                                                       classifier_optimizer_list[1:], train_dloader[1:]):
    #     for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
    #         if idx > batch_per_epoch:
    #             break
    #         image_s_i = image_s_i.cuda()
    #         label_s_i = label_s_i.long().cuda()
    #
    #         # TRAIN PHASE 0
    #         opt_g.zero_grad()
    #         opt_hs_i.zero_grad()
    #
    #         output_hs_i = hs_i(g_model(image_s_i))
    #         loss_class_src_i = softmaxcrossEntropy(output_hs_i, label_s_i)
    #         loss_class_src_lst.append(loss_class_src_i)
    #
    #         loss_class_src_i.backward()
    #         opt_g.step()
    #         opt_hs_i.step()
    #
    # i = -1
    # dloader_trg = train_dloader[0]
    # dloader_src_1 = train_dloader[1]
    # dloader_src_2 = train_dloader[2]
    # dloader_src_3 = train_dloader[3]
    # dloader_src_4 = train_dloader[4]
    # dloader_src_5 = train_dloader[5]
    # for (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
    #         image_s_4, label_s_4), (image_s_5, label_s_5) \
    #         in zip(dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
    #     # for (image_s_1, label_s_1), (image_s_2, label_s_2) \
    #     #         in zip(dloader_src_1, dloader_src_2):
    #     i += 1
    #     if i > batch_per_epoch:
    #         break
    #
    #     image_s_lst = [
    #         image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
    #     ]
    #     label_s_lst = [
    #         label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
    #         label_s_5.long().cuda()
    #     ]
    #
    #     # image_s_lst = [
    #     #     image_s_1.cuda(), image_s_2.cuda()
    #     # ]
    #     # label_s_lst = [
    #     #     label_s_1.long().cuda(), label_s_2.long().cuda()
    #     # ]
    #     # for opt_g in optimizer_g_lst[1:]:
    #     optimizer_g_lst[0].zero_grad()
    #     optimizer_dis.zero_grad()
    #     for opt_hs_i in classifier_optimizer_list[1:]:
    #         opt_hs_i.zero_grad()
    #     # loss_hs_combine_lst = []
    #     latent_s = []
    #
    #     # with torch.no_grad():
    #     for idx in range(len(image_s_lst)):
    #         latent_s_i = g_model_lst[0](image_s_lst[idx])
    #         latent_s.append(latent_s_i)
    #     latent_s = torch.cat(latent_s)
    #     out_hs = []
    #     for classifier in classifier_list[1:]:
    #         out_hs_i = classifier(latent_s)
    #         out_hs.append(softmax_fn(out_hs_i))
    #     out_dis_s = dis_model(latent_s)
    #     output_hs_combine_g_xs = hs_gx_C_gx(out_hs, out_dis_s)
    #     label_s = torch.cat(label_s_lst)
    #
    #     loss_hs_combine = crossEntropy_fn(output_hs_combine_g_xs, label_s, label_to_1hot=True)
    #     loss_hs_combine.backward()
    #     # for opt_g in optimizer_g_lst[1:]:
    #     #     opt_g.step()
    #     optimizer_g_lst[0].step()
    #     optimizer_dis.step()
    #     for opt_hs_i in classifier_optimizer_list[1:]:
    #         opt_hs_i.step()
    #
    # loss_mimic_ht_g_xt_mixed = 0
    # mean_confident = AverageMeter()
    #
    # i = -1
    # for image_t, label_t in dloader_trg:
    #     image_t = image_t.cuda()
    #     i+= 1
    #     if i > batch_per_epoch:
    #         break
    #     # label_t = label_t.long().cuda()
    #
    #     ## TRAIN PHASE 2: Gt + Ht mixup
    #     # optimizer_g_lst[0].zero_grad()
    #     classifier_optimizer_list[0].zero_grad()
    #
    #     with torch.no_grad():
    #         latent_trg = g_model_lst[0](image_t)
    #         output_dis_trg = dis_model(latent_trg)
    #         hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)
    #         hs_g_xt_conf, hs_g_xt_arg_max = hs_g_xt.max(1)
    #         hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)
    #         confidence_gate = torch.mean(hs_g_xt_conf.float()) * 1.2
    #         # if confidence_gate < 0.3:
    #         #     confidence_gate = 0.3
    #         # print("mean conf", torch.mean(hs_g_xt_conf.float()).item())
    #         hs_g_xt_mask = hs_g_xt_conf > confidence_gate
    #         if i % 500 == 0:
    #             print("confidence_gate", confidence_gate)
    #             print("torch.mean(hs_g_xt_mask.float())", torch.mean(hs_g_xt_mask.float()))
    #         image_t = image_t[hs_g_xt_mask, :]
    #         hs_g_xt_onehot = hs_g_xt_onehot[hs_g_xt_mask]
    #
    #         if image_t.size(0) < 1.0:
    #             continue
    #         lam = np.random.beta(2, 2)
    #         num_train_data = image_t.size(0)
    #         index = torch.randperm(num_train_data).cuda()
    #         mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
    #         # mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
    #         mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]
    #         # with torch.no_grad():
    #         latent_trg_mixed = g_model_lst[0](mixed_image)
    #     ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
    #     # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)
    #
    #     mean_confident.update(torch.mean(hs_g_xt_mask.float()))
    #     loss_mimic_ht_g_xt_mixed = torch.mean(
    #         torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
    #     # loss_mimic_ht_g_xt_mixed = torch.mean(
    #     #     torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
    #
    #     loss_mimic_ht_g_xt_mixed.backward()
    #     # optimizer_g_lst[0].step()
    #     classifier_optimizer_list[0].step()
    #
    # #### MIX Generator\
    # print("epoch {}:".format(epoch))
    # for n_s, l_s in zip(source_domains, loss_class_src_lst):
    #     print("\tloss_class_{}: \t{:.4f}".format(n_s, l_s.item()))
    #
    # print("\tloss_hs_combine_g_xs: {:.4f}".format(loss_hs_combine.item()))
    # print("MIMIC")
    # print("\tloss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed))
    # print("\tmean_confident_hs_combine_g_xt: {:.4f}".format(mean_confident.avg))
    # print("\tconfidence_gate: {:.4f}".format(confidence_gate))
    # acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
    # acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
    # acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
    # print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
    # print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
    # print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

    ###### print gan_loss_real
    # print("loss_gan_real: {:.4f}".format(loss_gan_real))

    # # source_domain_num = len(train_dloader_list[1:])
    # print(
    #     "train_multi_g_phase_4_separate_1_1_filter_3, mixup Ht, remove Gan, Gs_i, Gt, C and Hs_i train through hs_combine, update G_combine by Gs only, before train gt")
    # for g_model in g_model_lst:
    #     g_model.train()
    # dis_model.train()
    # # gan_model.train()
    # for hs_i in classifier_list:
    #     hs_i.train()
    # num_src_domain = len(source_domains)
    #
    # loss_class_src_lst = []
    # for g_model, opt_g, hs_i, opt_hs_i, dataLoader in zip(g_model_lst[1:], optimizer_g_lst[1:], classifier_list[1:],
    #                                                       classifier_optimizer_list[1:], train_dloader[1:]):
    #     for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
    #         if idx > batch_per_epoch:
    #             break
    #         image_s_i = image_s_i.cuda()
    #         label_s_i = label_s_i.long().cuda()
    #
    #         # TRAIN PHASE 0
    #         opt_g.zero_grad()
    #         opt_hs_i.zero_grad()
    #
    #         output_hs_i = hs_i(g_model(image_s_i))
    #         loss_class_src_i = softmaxcrossEntropy(output_hs_i, label_s_i)
    #         loss_class_src_lst.append(loss_class_src_i)
    #
    #         loss_class_src_i.backward()
    #         opt_g.step()
    #         opt_hs_i.step()
    #
    # # domain_weight = [1 / num_src_domain] * num_src_domain
    # # domain_weight.insert(0, 0)
    # domain_weight = [1 / (num_src_domain + 1)] * (num_src_domain + 1)
    # federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd, update_all=False)
    #
    # loss_mimic_ht_g_xt_mixed = 0
    # mean_confident = AverageMeter()
    # i = -1
    # dloader_trg = train_dloader[0]
    # dloader_src_1 = train_dloader[1]
    # dloader_src_2 = train_dloader[2]
    # dloader_src_3 = train_dloader[3]
    # dloader_src_4 = train_dloader[4]
    # dloader_src_5 = train_dloader[5]
    # for (image_t, label_t), (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
    #         image_s_4, label_s_4), (image_s_5, label_s_5) \
    #         in zip(dloader_trg, dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
    #     # for (image_s_1, label_s_1), (image_s_2, label_s_2) \
    #     #         in zip(dloader_src_1, dloader_src_2):
    #     i += 1
    #     if i > batch_per_epoch:
    #         break
    #
    #     image_s_lst = [
    #         image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
    #     ]
    #     label_s_lst = [
    #         label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
    #         label_s_5.long().cuda()
    #     ]
    #
    #     # image_s_lst = [
    #     #     image_s_1.cuda(), image_s_2.cuda()
    #     # ]
    #     # label_s_lst = [
    #     #     label_s_1.long().cuda(), label_s_2.long().cuda()
    #     # ]
    #     # for opt_g in optimizer_g_lst[1:]:
    #     # optimizer_g_lst[0].zero_grad()
    #     optimizer_dis.zero_grad()
    #     for opt_hs_i in classifier_optimizer_list[1:]:
    #         opt_hs_i.zero_grad()
    #     # loss_hs_combine_lst = []
    #     latent_s = []
    #
    #     with torch.no_grad():
    #         for idx in range(len(image_s_lst)):
    #             latent_s_i = g_model_lst[0](image_s_lst[idx])
    #             latent_s.append(latent_s_i)
    #         latent_s = torch.cat(latent_s)
    #     out_hs = []
    #     for classifier in classifier_list[1:]:
    #         out_hs_i = classifier(latent_s)
    #         out_hs.append(softmax_fn(out_hs_i))
    #     out_dis_s = dis_model(latent_s)
    #     output_hs_combine_g_xs = hs_gx_C_gx(out_hs, out_dis_s)
    #     label_s = torch.cat(label_s_lst)
    #
    #     loss_hs_combine = crossEntropy_fn(output_hs_combine_g_xs, label_s, label_to_1hot=True)
    #     loss_hs_combine.backward()
    #     # for opt_g in optimizer_g_lst[1:]:
    #     #     opt_g.step()
    #     # optimizer_g_lst[0].step()
    #     optimizer_dis.step()
    #     for opt_hs_i in classifier_optimizer_list[1:]:
    #         opt_hs_i.step()
    #
    #     optimizer_g_lst[0].zero_grad()
    #     classifier_optimizer_list[0].zero_grad()
    #
    #     with torch.no_grad():
    #         latent_trg = g_model_lst[0](image_t)
    #         output_dis_trg = dis_model(latent_trg)
    #         hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)
    #         hs_g_xt_conf, hs_g_xt_arg_max = hs_g_xt.max(1)
    #         hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)
    #         confidence_gate = torch.mean(hs_g_xt_conf.float()) * 1.2
    #         # if confidence_gate < 0.3:
    #         #     confidence_gate = 0.3
    #         # print("mean conf", torch.mean(hs_g_xt_conf.float()).item())
    #         hs_g_xt_mask = hs_g_xt_conf > confidence_gate
    #         if i % 500 == 0:
    #             print("confidence_gate", confidence_gate)
    #             print("torch.mean(hs_g_xt_mask.float())", torch.mean(hs_g_xt_mask.float()))
    #         image_t = image_t[hs_g_xt_mask, :]
    #         hs_g_xt_onehot = hs_g_xt_onehot[hs_g_xt_mask]
    #
    #         if image_t.size(0) < 1.0:
    #             continue
    #         lam = np.random.beta(2, 2)
    #         num_train_data = image_t.size(0)
    #         index = torch.randperm(num_train_data).cuda()
    #         mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
    #         # mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
    #         mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]
    #         # with torch.no_grad():
    #     latent_trg_mixed = g_model_lst[0](mixed_image)
    #     ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
    #     # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)
    #
    #     mean_confident.update(torch.mean(hs_g_xt_mask.float()))
    #     loss_mimic_ht_g_xt_mixed = torch.mean(
    #         torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
    #     # loss_mimic_ht_g_xt_mixed = torch.mean(
    #     #     torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
    #
    #     loss_mimic_ht_g_xt_mixed.backward()
    #     optimizer_g_lst[0].step()
    #     classifier_optimizer_list[0].step()
    #
    # #### MIX Generator\
    # print("epoch {}:".format(epoch))
    # for n_s, l_s in zip(source_domains, loss_class_src_lst):
    #     print("\tloss_class_{}: \t{:.4f}".format(n_s, l_s.item()))
    #
    # print("\tloss_hs_combine_g_xs: {:.4f}".format(loss_hs_combine.item()))
    # print("MIMIC")
    # print("\tloss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed))
    # print("\tmean_confident_hs_combine_g_xt: {:.4f}".format(mean_confident.avg))
    # print("\tconfidence_gate: {:.4f}".format(confidence_gate))

    # source_domain_num = len(train_dloader_list[1:])
    # print(
    #     "train_multi_g_phase_4_separate_1_1_filter_2, mixup Ht, remove Gan, Gs_i, Gt, C and Hs_i train through hs_combine, update G_combine by Gs only, before train gt")
    # for g_model in g_model_lst:
    #     g_model.train()
    # dis_model.train()
    # # gan_model.train()
    # for hs_i in classifier_list:
    #     hs_i.train()
    # num_src_domain = len(source_domains)
    #
    # loss_class_src_lst = []
    # for g_model, opt_g, hs_i, opt_hs_i, dataLoader in zip(g_model_lst[1:], optimizer_g_lst[1:], classifier_list[1:],
    #                                                       classifier_optimizer_list[1:], train_dloader[1:]):
    #     for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
    #         if idx > batch_per_epoch:
    #             break
    #         image_s_i = image_s_i.cuda()
    #         label_s_i = label_s_i.long().cuda()
    #
    #         # TRAIN PHASE 0
    #         opt_g.zero_grad()
    #         opt_hs_i.zero_grad()
    #
    #         output_hs_i = hs_i(g_model(image_s_i))
    #         loss_class_src_i = softmaxcrossEntropy(output_hs_i, label_s_i)
    #         loss_class_src_lst.append(loss_class_src_i)
    #
    #         loss_class_src_i.backward()
    #         opt_g.step()
    #         opt_hs_i.step()
    #
    # # domain_weight = [1 / num_src_domain] * num_src_domain
    # # domain_weight.insert(0, 0)
    # domain_weight = [1 / (num_src_domain + 1)] * (num_src_domain + 1)
    # federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd, update_all=False)
    #
    # i = -1
    # dloader_trg = train_dloader[0]
    # dloader_src_1 = train_dloader[1]
    # dloader_src_2 = train_dloader[2]
    # dloader_src_3 = train_dloader[3]
    # dloader_src_4 = train_dloader[4]
    # dloader_src_5 = train_dloader[5]
    # for (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
    #         image_s_4, label_s_4), (image_s_5, label_s_5) \
    #         in zip(dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
    #     # for (image_s_1, label_s_1), (image_s_2, label_s_2) \
    #     #         in zip(dloader_src_1, dloader_src_2):
    #     i += 1
    #     if i > batch_per_epoch:
    #         break
    #
    #     image_s_lst = [
    #         image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
    #     ]
    #     label_s_lst = [
    #         label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
    #         label_s_5.long().cuda()
    #     ]
    #
    #     # image_s_lst = [
    #     #     image_s_1.cuda(), image_s_2.cuda()
    #     # ]
    #     # label_s_lst = [
    #     #     label_s_1.long().cuda(), label_s_2.long().cuda()
    #     # ]
    #     # for opt_g in optimizer_g_lst[1:]:
    #     optimizer_g_lst[0].zero_grad()
    #     optimizer_dis.zero_grad()
    #     for opt_hs_i in classifier_optimizer_list[1:]:
    #         opt_hs_i.zero_grad()
    #     # loss_hs_combine_lst = []
    #     latent_s = []
    #
    #     # with torch.no_grad():
    #     for idx in range(len(image_s_lst)):
    #         latent_s_i = g_model_lst[0](image_s_lst[idx])
    #         latent_s.append(latent_s_i)
    #     latent_s = torch.cat(latent_s)
    #     out_hs = []
    #     for classifier in classifier_list[1:]:
    #         out_hs_i = classifier(latent_s)
    #         out_hs.append(softmax_fn(out_hs_i))
    #     out_dis_s = dis_model(latent_s)
    #     output_hs_combine_g_xs = hs_gx_C_gx(out_hs, out_dis_s)
    #     label_s = torch.cat(label_s_lst)
    #
    #     loss_hs_combine = crossEntropy_fn(output_hs_combine_g_xs, label_s, label_to_1hot=True)
    #     loss_hs_combine.backward()
    #     # for opt_g in optimizer_g_lst[1:]:
    #     #     opt_g.step()
    #     optimizer_g_lst[0].step()
    #     optimizer_dis.step()
    #     for opt_hs_i in classifier_optimizer_list[1:]:
    #         opt_hs_i.step()
    #
    # loss_mimic_ht_g_xt_mixed = 0
    # mean_confident = AverageMeter()
    #
    # # if epoch > 2:
    #     # confidence_gate = (0.5 - 0.2) * (epoch / 60) + 0.2
    #     # if confidence_gate > 0.5:
    #     #     confidence_gate = 0.5
    #
    # i = -1
    # for image_t, label_t in dloader_trg:
    #     image_t = image_t.cuda()
    #     i+= 1
    #     if i > batch_per_epoch:
    #         break
    #     # label_t = label_t.long().cuda()
    #
    #     ## TRAIN PHASE 2: Gt + Ht mixup
    #     optimizer_g_lst[0].zero_grad()
    #     classifier_optimizer_list[0].zero_grad()
    #
    #     with torch.no_grad():
    #         latent_trg = g_model_lst[0](image_t)
    #         output_dis_trg = dis_model(latent_trg)
    #         hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)
    #         hs_g_xt_conf, hs_g_xt_arg_max = hs_g_xt.max(1)
    #         hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)
    #         confidence_gate = torch.mean(hs_g_xt_conf.float()) * 1.2
    #         # if confidence_gate < 0.3:
    #         #     confidence_gate = 0.3
    #         # print("mean conf", torch.mean(hs_g_xt_conf.float()).item())
    #         hs_g_xt_mask = hs_g_xt_conf > confidence_gate
    #         if i % 500 == 0:
    #             print("confidence_gate", confidence_gate)
    #             print("torch.mean(hs_g_xt_mask.float())", torch.mean(hs_g_xt_mask.float()))
    #         image_t = image_t[hs_g_xt_mask, :]
    #         hs_g_xt_onehot = hs_g_xt_onehot[hs_g_xt_mask]
    #
    #         if image_t.size(0) < 1.0:
    #             continue
    #         lam = np.random.beta(2, 2)
    #         num_train_data = image_t.size(0)
    #         index = torch.randperm(num_train_data).cuda()
    #         mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
    #         # mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
    #         mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]
    #         # with torch.no_grad():
    #     latent_trg_mixed = g_model_lst[0](mixed_image)
    #     ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
    #     # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)
    #
    #     mean_confident.update(torch.mean(hs_g_xt_mask.float()))
    #     loss_mimic_ht_g_xt_mixed = torch.mean(
    #         torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
    #     # loss_mimic_ht_g_xt_mixed = torch.mean(
    #     #     torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
    #
    #     loss_mimic_ht_g_xt_mixed.backward()
    #     optimizer_g_lst[0].step()
    #     classifier_optimizer_list[0].step()
    #
    # #### MIX Generator\
    # print("epoch {}:".format(epoch))
    # for n_s, l_s in zip(source_domains, loss_class_src_lst):
    #     print("\tloss_class_{}: \t{:.4f}".format(n_s, l_s.item()))
    #
    # print("\tloss_hs_combine_g_xs: {:.4f}".format(loss_hs_combine.item()))
    # print("MIMIC")
    # print("\tloss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed))
    # print("\tmean_confident_hs_combine_g_xt: {:.4f}".format(mean_confident.avg))
    # print("\tconfidence_gate: {:.4f}".format(confidence_gate))
    # # acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
    # # acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
    # # acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
    # # print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
    # # print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
    # # print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))
    #
    # ###### print gan_loss_real
    # # print("loss_gan_real: {:.4f}".format(loss_gan_real))


def train_multi_g_phase_4_separate_1_1_c(train_dloader, g_model_lst, classifier_list, dis_model, gan_model,
                                         optimizer_g_lst,
                                         optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                         num_classes, source_domains, batchnorm_mmd, batch_per_epoch):
    # source_domain_num = len(train_dloader_list[1:])
    print(
        "train_multi_g_phase_4_separate_1_1_c, mixup Ht, remove Gan, Gs_i, Gt, C and Hs_i train through hs_combine and real label dis, update G_combine by Gs only, before train gt")
    for g_model in g_model_lst:
        g_model.train()
    dis_model.train()
    # gan_model.train()
    for hs_i in classifier_list:
        hs_i.train()
    num_src_domain = len(source_domains)

    dloader_trg = train_dloader[0]
    loss_class_src_lst = []
    for g_model, opt_g, hs_i, opt_hs_i, dataLoader in zip(g_model_lst[1:], optimizer_g_lst[1:], classifier_list[1:],
                                                          classifier_optimizer_list[1:], train_dloader[1:]):
        for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
            if idx > batch_per_epoch:
                break
            image_s_i = image_s_i.cuda()
            label_s_i = label_s_i.long().cuda()

            # TRAIN PHASE 0
            opt_g.zero_grad()
            opt_hs_i.zero_grad()

            output_hs_i = hs_i(g_model(image_s_i))
            loss_class_src_i = softmaxcrossEntropy(output_hs_i, label_s_i)
            loss_class_src_lst.append(loss_class_src_i)

            loss_class_src_i.backward()
            opt_g.step()
            opt_hs_i.step()

    domain_weight = [1 / num_src_domain] * num_src_domain
    domain_weight.insert(0, 0)
    federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)

    i = -1
    dloader_src_1 = train_dloader[1]
    dloader_src_2 = train_dloader[2]
    dloader_src_3 = train_dloader[3]
    dloader_src_4 = train_dloader[4]
    dloader_src_5 = train_dloader[5]
    for (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        i += 1
        if i == batch_per_epoch:
            break

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]

        optimizer_dis.zero_grad()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.zero_grad()
        # loss_hs_combine_lst = []
        latent_s = []

        dis_model_label = []
        with torch.no_grad():
            for idx in range(len(image_s_lst)):
                latent_s_i = g_model_lst[idx + 1](image_s_lst[idx])
                latent_s.append(latent_s_i)
                dis_model_label.append(torch.ones(latent_s_i.size(0)) * idx)
            latent_s = torch.cat(latent_s)
            dis_model_label = torch.cat(dis_model_label).long().cuda()

        out_hs = []
        for classifier in classifier_list[1:]:
            out_hs_i = classifier(latent_s)
            out_hs.append(softmax_fn(out_hs_i))
        out_dis_s = dis_model(latent_s)
        output_hs_combine_g_xs = hs_gx_C_gx(out_hs, out_dis_s)
        label_s = torch.cat(label_s_lst)

        loss_hs_combine = crossEntropy_fn(output_hs_combine_g_xs, label_s, label_to_1hot=True)
        #         loss_hs_combine.backward()
        loss_dis_model = softmaxcrossEntropy(out_dis_s, dis_model_label)
        #         loss_dis_model.backward()

        loss_dis_sum = sum([loss_hs_combine, loss_dis_model])
        loss_dis_sum.backward()

        optimizer_dis.step()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.step()

    loss_mimic_ht_g_xt_mixed = 0
    mean_confident = AverageMeter()
    for image_t, label_t in dloader_trg:
        image_t = image_t.cuda()
        # label_t = label_t.long().cuda()

        ## TRAIN PHASE 2: Gt + Ht mixup
        # optimizer_g_lst[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()

        with torch.no_grad():
            latent_trg = g_model_lst[0](image_t)
            output_dis_trg = dis_model(latent_trg)
            hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)
            hs_g_xt_max_by_class, hs_g_xt_arg_max = hs_g_xt.max(1)
            hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)

            lam = np.random.beta(2, 2)
            batch_size = image_t.size(0)
            index = torch.randperm(batch_size).cuda()
            mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
            # mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
            mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]
            # with torch.no_grad():
            latent_trg_mixed = g_model_lst[0](mixed_image)
        ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
        # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)

        confidence_gate = (0.95 - 0.6) * (epoch / 60) + 0.6
        if confidence_gate > 0.95:
            confidence_gate = 0.95
        hs_g_xt_mask = (hs_g_xt_max_by_class > confidence_gate).float().cuda()
        mean_confident.update(torch.mean(hs_g_xt_mask))
        loss_mimic_ht_g_xt_mixed = torch.mean(
            hs_g_xt_mask * torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
        # loss_mimic_ht_g_xt_mixed = torch.mean(
        #     torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))

        loss_mimic_ht_g_xt_mixed.backward()
        # optimizer_g_lst[0].step()
        classifier_optimizer_list[0].step()

    #### MIX Generator\
    print("epoch {}:".format(epoch))
    for n_s, l_s in zip(source_domains, loss_class_src_lst):
        print("\tloss_class_{}: \t{:.4f}".format(n_s, l_s.item()))

    print("\tloss_hs_combine_g_xs: {:.4f}".format(loss_hs_combine.item()))
    print("MIMIC")
    print("\tloss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed.item()))
    print("\tmean_confident_hs_combine_g_xt: {:.4f}".format(mean_confident.avg))
    # acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
    # acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
    # acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
    # print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
    # print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
    # print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

    ###### print gan_loss_real
    # print("loss_gan_real: {:.4f}".format(loss_gan_real))


def train_multi_g_phase_4_separate_1_1_group(train_dloader, g_model_lst, classifier_list, dis_model, gan_model,
                                             optimizer_g_lst,
                                             optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                             num_classes, source_domains, batchnorm_mmd, batch_per_epoch):
    # source_domain_num = len(train_dloader_list[1:])
    print(
        "train_multi_g_phase_4_separate_1_1_group, mixup Ht, remove Gan, Gs_i, Gt, C and Hs_i train through hs_combine, update G_combine by Gs only, before train gt")
    for g_model in g_model_lst:
        g_model.train()
    dis_model.train()
    # gan_model.train()
    for hs_i in classifier_list:
        hs_i.train()
    num_src_domain = len(source_domains)

    dloader_trg = train_dloader[0]
    loss_class_src_lst = []
    for g_model, opt_g, hs_i, opt_hs_i, dataLoader in zip(g_model_lst[1:], optimizer_g_lst[1:], classifier_list[1:],
                                                          classifier_optimizer_list[1:], train_dloader[1:]):
        for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
            if idx > batch_per_epoch:
                break
            image_s_i = image_s_i.cuda()
            label_s_i = label_s_i.long().cuda()

            # TRAIN PHASE 0
            opt_g.zero_grad()
            opt_hs_i.zero_grad()

            output_hs_i = hs_i(g_model(image_s_i))
            loss_class_src_i = softmaxcrossEntropy(output_hs_i, label_s_i)
            loss_class_src_lst.append(loss_class_src_i)

            loss_class_src_i.backward()
            opt_g.step()
            opt_hs_i.step()

    domain_weight = [1 / num_src_domain] * num_src_domain
    domain_weight.insert(0, 0)
    federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)

    confidence_gate = (0.95 - 0.6) * (epoch / 60) + 0.6
    if confidence_gate > 0.95:
        confidence_gate = 0.95
    i = -1
    dloader_src_1 = train_dloader[1]
    dloader_src_2 = train_dloader[2]
    dloader_src_3 = train_dloader[3]
    dloader_src_4 = train_dloader[4]
    dloader_src_5 = train_dloader[5]
    for (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        i += 1
        if i == batch_per_epoch:
            break

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]

        optimizer_dis.zero_grad()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.zero_grad()
        # loss_hs_combine_lst = []
        latent_s = []

        with torch.no_grad():
            for idx in range(len(image_s_lst)):
                latent_s_i = g_model_lst[idx + 1](image_s_lst[idx])
                latent_s.append(latent_s_i)
            latent_s = torch.cat(latent_s)
        out_hs = []
        for classifier in classifier_list[1:]:
            out_hs_i = classifier(latent_s)
            out_hs.append(softmax_fn(out_hs_i))
        out_dis_s = dis_model(latent_s)
        output_hs_combine_g_xs = hs_gx_C_gx(out_hs, out_dis_s)
        label_s = torch.cat(label_s_lst)

        loss_hs_combine = crossEntropy_fn(output_hs_combine_g_xs, label_s, label_to_1hot=True)
        loss_hs_combine.backward()
        optimizer_dis.step()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.step()

    mean_confident = AverageMeter()
    for image_t, label_t in dloader_trg:
        image_t = image_t.cuda()
        # label_t = label_t.long().cuda()

        ## TRAIN PHASE 2: Gt + Ht mixup
        # optimizer_g_lst[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()

        with torch.no_grad():
            latent_trg = g_model_lst[0](image_t)
            output_dis_trg = dis_model(latent_trg)
            hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)
            hs_g_xt_max_by_class, hs_g_xt_arg_max = hs_g_xt.max(1)
            hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)

        lam = np.random.beta(2, 2)
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        # mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
        mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]
        with torch.no_grad():
            latent_trg_mixed = g_model_lst[0](mixed_image)
        ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
        # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)

        hs_g_xt_mask = (hs_g_xt_max_by_class > confidence_gate).float().cuda()
        mean_confident.update(torch.mean(hs_g_xt_mask))
        loss_mimic_ht_g_xt_mixed = torch.mean(
            hs_g_xt_mask * torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
        # loss_mimic_ht_g_xt_mixed = torch.mean(
        #     torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))

        loss_mimic_ht_g_xt_mixed.backward()
        # optimizer_g_lst[0].step()
        classifier_optimizer_list[0].step()

    #### MIX Generator\
    print("epoch {}:".format(epoch))
    for n_s, l_s in zip(source_domains, loss_class_src_lst):
        print("\tloss_class_{}: \t{:.4f}".format(n_s, l_s.item()))

    print("\tloss_hs_combine_g_xs: {:.4f}".format(loss_hs_combine.item()))
    print("MIMIC")
    print("\tloss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed.item()))
    print("\tmean_confident_hs_combine_g_xt: {:.4f}".format(mean_confident.avg))
    # acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
    # acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
    # acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
    # print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
    # print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
    # print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

    ###### print gan_loss_real
    # print("loss_gan_real: {:.4f}".format(loss_gan_real))


def train_multi_g_phase_4_separate_1_1_mix(train_dloader, g_model_lst, classifier_list, dis_model, gan_model,
                                           optimizer_g_lst,
                                           optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                           num_classes, source_domains, batchnorm_mmd, batch_per_epoch):
    # source_domain_num = len(train_dloader_list[1:])
    print(
        "train_multi_g_phase_4_separate_1_1, mixup Ht, remove Gan, Gs_i, Gt, C and Hs_i train through hs_combine, update G_combine by Gs only, before train gt")
    for g_model in g_model_lst:
        g_model.train()
    dis_model.train()
    # gan_model.train()
    for hs_i in classifier_list:
        hs_i.train()
    num_src_domain = len(source_domains)

    dloader_trg = train_dloader[0]
    loss_class_src_lst = []
    for g_model, opt_g, hs_i, opt_hs_i, dataLoader in zip(g_model_lst[1:], optimizer_g_lst[1:], classifier_list[1:],
                                                          classifier_optimizer_list[1:], train_dloader[1:]):
        for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
            if idx > batch_per_epoch:
                break
            image_s_i = image_s_i.cuda()
            label_s_i = label_s_i.long().cuda()

            # TRAIN PHASE 0
            opt_g.zero_grad()
            opt_hs_i.zero_grad()

            output_hs_i = hs_i(g_model(image_s_i))
            loss_class_src_i = softmaxcrossEntropy(output_hs_i, label_s_i)
            loss_class_src_lst.append(loss_class_src_i)

            loss_class_src_i.backward()
            opt_g.step()
            opt_hs_i.step()

    domain_weight = [1 / num_src_domain] * num_src_domain
    domain_weight.insert(0, 0)
    federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)

    i = -1
    dloader_src_1 = train_dloader[1]
    dloader_src_2 = train_dloader[2]
    dloader_src_3 = train_dloader[3]
    dloader_src_4 = train_dloader[4]
    dloader_src_5 = train_dloader[5]
    for (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        i += 1
        if i == batch_per_epoch:
            break

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]

        optimizer_dis.zero_grad()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.zero_grad()
        # loss_hs_combine_lst = []
        latent_s = []

        with torch.no_grad():
            for idx in range(len(image_s_lst)):
                latent_s_i = g_model_lst[idx + 1](image_s_lst[idx])
                latent_s.append(latent_s_i)
            latent_s = torch.cat(latent_s)
        out_hs = []
        for classifier in classifier_list[1:]:
            out_hs_i = classifier(latent_s)
            out_hs.append(softmax_fn(out_hs_i))
        out_dis_s = dis_model(latent_s)
        output_hs_combine_g_xs = hs_gx_C_gx(out_hs, out_dis_s)
        label_s = torch.cat(label_s_lst)

        loss_hs_combine = crossEntropy_fn(output_hs_combine_g_xs, label_s, label_to_1hot=True)
        loss_hs_combine.backward()
        optimizer_dis.step()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.step()

    confidence_gate = (0.95 - 0.6) * (epoch / 60) + 0.6
    if confidence_gate > 0.95:
        confidence_gate = 0.95

    mean_confident = AverageMeter()
    for (image_t, label_t), (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_trg, dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        # for image_t, label_t in dloader_trg:
        image_t = image_t.cuda()
        batch_size = image_t.size(0)

        # label_t = label_t.long().cuda()

        ## TRAIN PHASE 2: Gt + Ht mixup
        # optimizer_g_lst[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()

        with torch.no_grad():
            latent_trg = g_model_lst[0](image_t)
            output_dis_trg = dis_model(latent_trg)
            hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)
            hs_g_xt_max_by_class, hs_g_xt_arg_max = hs_g_xt.max(1)
            hs_g_xt_mask = hs_g_xt_max_by_class > confidence_gate
            hs_g_xt_arg_max = hs_g_xt_arg_max[hs_g_xt_mask]  # keep confident data to train only
            image_t = image_t[hs_g_xt_mask]

            image_lst = [
                image_t, image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
            ]
            label_lst = [hs_g_xt_arg_max, label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(),
                         label_s_4.long().cuda(), label_s_5.long().cuda()
                         ]

            label_all = torch.cat(label_lst)
            label_all = label_to_1hot(label_all, num_classes)

        lam = np.random.beta(2, 2)
        img_all = torch.cat(image_lst)

        num_data_mix = img_all.size(0)
        index = torch.randperm(num_data_mix).cuda()
        mixed_image = lam * img_all + (1 - lam) * img_all[index, :]
        # mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
        mixed_label = lam * label_all + (1 - lam) * label_all[index, :]
        with torch.no_grad():
            latent_mix = []
            for img_mix_idx in range(0, num_data_mix, batch_size):
                img_mix_i = mixed_image[img_mix_idx: img_mix_idx + batch_size]
                latent_mix_i = g_model_lst[0](img_mix_i)
                latent_mix.append(latent_mix_i)
            latent_mix = torch.cat(latent_mix)
        ht_g_xt_mixed = classifier_list[0](latent_mix)
        # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)

        mean_confident.update(torch.mean(hs_g_xt_mask.float()))
        loss_mimic_ht_g_xt_mixed = torch.mean(
            torch.sum(-1.0 * mixed_label * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
        # loss_mimic_ht_g_xt_mixed = torch.mean(
        #     torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))

        loss_mimic_ht_g_xt_mixed.backward()
        # optimizer_g_lst[0].step()
        classifier_optimizer_list[0].step()

    #### MIX Generator\
    print("epoch {}:".format(epoch))
    for n_s, l_s in zip(source_domains, loss_class_src_lst):
        print("\tloss_class_{}: \t{:.4f}".format(n_s, l_s.item()))

    print("\tloss_hs_combine_g_xs: {:.4f}".format(loss_hs_combine.item()))
    print("MIMIC")
    print("\tloss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed.item()))
    print("\tmean_confident_hs_combine_g_xt: {:.4f}".format(mean_confident.avg))
    # acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
    # acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
    # acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
    # print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
    # print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
    # print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

    ###### print gan_loss_real
    # print("loss_gan_real: {:.4f}".format(loss_gan_real))


def train_multi_g_phase_4_separate_1_2(train_dloader, g_model_lst, classifier_list, dis_model, gan_model,
                                       optimizer_g_lst,
                                       optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                       num_classes, source_domains_name, batchnorm_mmd, batch_per_epoch):
    # source_domain_num = len(train_dloader_list[1:])
    print("updated")
    print(
        "train_multi_g_phase_4_separate_1_2, mixup Ht, remove Gan, Gs_i, Gt, C and Hs_i train through hs_combine, update G_combine by Gs only, before train gt, update G_combine by Gs and Gt, after all")
    for g_model in g_model_lst:
        g_model.train()
    dis_model.train()
    # gan_model.train()
    for hs_i in classifier_list:
        hs_i.train()
    num_src_domain = 5

    loss_class_src_lst = []
    for g_model, opt_g, hs_i, opt_hs_i, dataLoader in zip(g_model_lst[1:], optimizer_g_lst[1:], classifier_list[1:],
                                                          classifier_optimizer_list[1:], train_dloader[1:]):
        for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
            if idx > batch_per_epoch:
                break
            image_s_i = image_s_i.cuda()
            label_s_i = label_s_i.long().cuda()

            # TRAIN PHASE 0
            opt_g.zero_grad()
            opt_hs_i.zero_grad()

            output_hs_i = hs_i(g_model(image_s_i))
            loss_class_src_i = softmaxcrossEntropy(output_hs_i, label_s_i)
            loss_class_src_lst.append(loss_class_src_i)

            loss_class_src_i.backward()
            opt_g.step()
            opt_hs_i.step()

    i = -1
    dloader_trg = train_dloader[0]
    dloader_src_1 = train_dloader[1]
    dloader_src_2 = train_dloader[2]
    dloader_src_3 = train_dloader[3]
    dloader_src_4 = train_dloader[4]
    dloader_src_5 = train_dloader[5]
    for (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        i += 1
        if i == batch_per_epoch:
            break

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]
        optimizer_dis.zero_grad()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.zero_grad()
        # loss_hs_combine_lst = []
        latent_s = []

        with torch.no_grad():
            for idx in range(len(image_s_lst)):
                latent_s_i = g_model_lst[idx + 1](image_s_lst[idx])
                latent_s.append(latent_s_i)
            latent_s = torch.cat(latent_s)
        out_hs = []
        for classifier in classifier_list[1:]:
            out_hs_i = classifier(latent_s)
            out_hs.append(softmax_fn(out_hs_i))
        out_dis_s = dis_model(latent_s)
        output_hs_combine_g_xs = hs_gx_C_gx(out_hs, out_dis_s)
        label_s = torch.cat(label_s_lst)

        loss_hs_combine = crossEntropy_fn(output_hs_combine_g_xs, label_s, label_to_1hot=True)
        loss_hs_combine.backward()
        optimizer_dis.step()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.step()

    ### PHASE 3: train Gan, Gs, Gt + GAN only #####
    # real_label_gan = 1.
    # fake_label_gan = 0.

    domain_weight = [1 / num_src_domain] * num_src_domain
    domain_weight.insert(0, 0)
    federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)

    ## TRAIN GAN
    # i = -1
    # for (image_t, label_t) in dloader_trg:
    #     i += 1
    #     if i > batch_per_epoch:
    #         break
    #     image_t = image_t.cuda()
    #     gan_model.zero_grad()
    #     with torch.no_grad():
    #         latent_trg = g_model_lst[0](image_t)
    #         latent_t_fake_lst = []
    #         for idx in range(num_src_domain):
    #             latent_t_idx = g_model_lst[idx + 1](image_t)
    #             # latent_s_idx = g_model_lst[idx + 1](image_s_lst[idx][:4])
    #             latent_t_fake_lst.append(latent_t_idx)
    #
    #         latent_t_fake = torch.cat(latent_t_fake_lst, dim=0)
    #         latent_trg_src = torch.cat([latent_t_fake, latent_trg], dim=0)
    #         gt_gan_src_real = torch.zeros(latent_t_fake.size(0)).float().cuda()
    #         gt_gan_trg_real = torch.ones(latent_trg.size(0)).float().cuda()
    #
    #     out_gan_real = torch.sigmoid(gan_model(latent_trg_src)).view(-1)
    #     gt_gan_real = torch.cat([gt_gan_src_real, gt_gan_trg_real], dim=0)
    #     loss_gan_real = gan_loss_fn(out_gan_real, gt_gan_real)
    #     loss_gan_real.backward()
    #     optimizer_gan.step()
    #
    #     g_model_lst[0].zero_grad()
    #     latent_trg = g_model_lst[0](image_t)
    #     out_gan_fake = torch.sigmoid(gan_model(latent_trg)).view(-1)
    #     gt_gan_fake = torch.zeros(latent_trg.size(0)).float().cuda()
    #     loss_gan_fake = gan_loss_fn(out_gan_fake, gt_gan_fake)
    #     loss_gan_fake.backward()
    #     optimizer_g_lst[0].step()

    mean_confident = AverageMeter()
    for i, (image_t, _) in enumerate(dloader_trg):
        if i > batch_per_epoch:
            break
        image_t = image_t.cuda()
        # label_t = label_t.long().cuda()

        ## TRAIN PHASE 2: Gt + Ht mixup
        optimizer_g_lst[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()

        with torch.no_grad():
            latent_trg = g_model_lst[1](image_t)
            output_dis_trg = dis_model(latent_trg)
            hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)
            hs_g_xt_max_by_class, hs_g_xt_arg_max = hs_g_xt.max(1)
            hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)

        lam = np.random.beta(2, 2)
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        # mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
        mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]

        latent_trg_mixed = g_model_lst[0](mixed_image)
        ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
        # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)

        confidence_gate = (0.7 - 0.3) * (epoch / 100) + 0.3
        if confidence_gate > 0.7:
            confidence_gate = 0.7
        hs_g_xt_mask = (hs_g_xt_max_by_class > confidence_gate).float().cuda()
        mean_confident.update(torch.mean(hs_g_xt_mask))
        loss_mimic_ht_g_xt_mixed = torch.mean(
            hs_g_xt_mask * torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
        # loss_mimic_ht_g_xt_mixed = torch.mean(
        #     torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))

        loss_mimic_ht_g_xt_mixed.backward()
        optimizer_g_lst[0].step()
        classifier_optimizer_list[0].step()

    # train time 2
    # domain_weight = [1 / (num_src_domain + 1)] * (num_src_domain + 1)
    # domain_weight.insert(0, domain_weight[-1] * 0.1)
    # federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)
    # train time 3
    domain_weight = [1 / (num_src_domain + 0.2)] * num_src_domain
    domain_weight.insert(0, domain_weight[-1] * 0.2)
    federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)

    # compute mmd:
    # latent_s = []
    # with torch.no_grad():
    #     for g_model, dataLoader in zip(g_model_lst[1:], train_dloader[1:]):
    #         latent_s_i = []
    #         for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
    #             if idx == 10:
    #                 break
    #             image_s_i = image_s_i.cuda()
    #             latent_s_i.append(g_model(image_s_i).view(batch_size, -1))
    #         latent_s.append(torch.cat(latent_s_i))
    # print("Compute MMD")
    # for i, s_n in enumerate(source_domains_name[:num_src_domain-1]):
    #     print(s_n)
    #     for j, s_n_j in enumerate(source_domains_name[i+1:]):
    #         print("\t{}: {}".format(s_n_j, compute_mmd(latent_s[i], latent_s[j])))

    #### MIX Generator\
    print("epoch {}:".format(epoch))
    for n_s, l_s in zip(source_domains_name, loss_class_src_lst):
        print("\tloss_class_{}: \t{:.4f}".format(n_s, l_s.item()))

    print("\tloss_hs_combine_g_xs: {:.4f}".format(loss_hs_combine.item()))
    print("MIMIC")
    print("\tloss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed.item()))
    print("\tmean_confident_hs_combine_g_xt: {:.4f}".format(mean_confident.avg))
    # acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
    # acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
    # acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
    # print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
    # print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
    # print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

    ###### print gan_loss_real
    # print("loss_gan_real: {:.4f}".format(loss_gan_real))

def train_multi_g_phase_4_separate_1_2_confi(train_dloader, g_model_lst, classifier_list, dis_model, gan_model,
                                       optimizer_g_lst,
                                       optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                       num_classes, source_domains_name, batchnorm_mmd, batch_per_epoch, confidence_gate):
    # source_domain_num = len(train_dloader_list[1:])
    print("updated")
    print(
        "train_multi_g_phase_4_separate_1_2_confi, mixup Ht, remove Gan, Gs_i, Gt, C and Hs_i train through hs_combine, update G_combine by Gs only, before train gt, update G_combine by Gs and Gt, after all")
    for g_model in g_model_lst:
        g_model.train()
    dis_model.train()
    # gan_model.train()
    for hs_i in classifier_list:
        hs_i.train()
    num_src_domain = 5

    loss_class_src_lst = []
    for g_model, opt_g, hs_i, opt_hs_i, dataLoader in zip(g_model_lst[1:], optimizer_g_lst[1:], classifier_list[1:],
                                                          classifier_optimizer_list[1:], train_dloader[1:]):
        for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
            if idx > batch_per_epoch:
                break
            image_s_i = image_s_i.cuda()
            label_s_i = label_s_i.long().cuda()

            # TRAIN PHASE 0
            opt_g.zero_grad()
            opt_hs_i.zero_grad()

            output_hs_i = hs_i(g_model(image_s_i))
            loss_class_src_i = softmaxcrossEntropy(output_hs_i, label_s_i)
            loss_class_src_lst.append(loss_class_src_i)

            loss_class_src_i.backward()
            opt_g.step()
            opt_hs_i.step()

    domain_weight = [1 / num_src_domain] * num_src_domain
    domain_weight.insert(0, 0)
    federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)

    i = -1
    dloader_trg = train_dloader[0]
    dloader_src_1 = train_dloader[1]
    dloader_src_2 = train_dloader[2]
    dloader_src_3 = train_dloader[3]
    dloader_src_4 = train_dloader[4]
    dloader_src_5 = train_dloader[5]
    for (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        i += 1
        if i == batch_per_epoch:
            break

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]
        optimizer_dis.zero_grad()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.zero_grad()
        # loss_hs_combine_lst = []
        latent_s = []

        with torch.no_grad():
            for idx in range(len(image_s_lst)):
                latent_s_i = g_model_lst[idx + 1](image_s_lst[idx])
                latent_s.append(latent_s_i)
            latent_s = torch.cat(latent_s)
        out_hs = []
        for classifier in classifier_list[1:]:
            out_hs_i = classifier(latent_s)
            out_hs.append(softmax_fn(out_hs_i))
        out_dis_s = dis_model(latent_s)
        output_hs_combine_g_xs = hs_gx_C_gx(out_hs, out_dis_s)
        label_s = torch.cat(label_s_lst)

        loss_hs_combine = crossEntropy_fn(output_hs_combine_g_xs, label_s, label_to_1hot=True)
        loss_hs_combine.backward()
        optimizer_dis.step()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.step()

    ### PHASE 3: train Gan, Gs, Gt + GAN only #####
    # real_label_gan = 1.
    # fake_label_gan = 0.

    ## TRAIN GAN
    # i = -1
    # for (image_t, label_t) in dloader_trg:
    #     i += 1
    #     if i > batch_per_epoch:
    #         break
    #     image_t = image_t.cuda()
    #     gan_model.zero_grad()
    #     with torch.no_grad():
    #         latent_trg = g_model_lst[0](image_t)
    #         latent_t_fake_lst = []
    #         for idx in range(num_src_domain):
    #             latent_t_idx = g_model_lst[idx + 1](image_t)
    #             # latent_s_idx = g_model_lst[idx + 1](image_s_lst[idx][:4])
    #             latent_t_fake_lst.append(latent_t_idx)
    #
    #         latent_t_fake = torch.cat(latent_t_fake_lst, dim=0)
    #         latent_trg_src = torch.cat([latent_t_fake, latent_trg], dim=0)
    #         gt_gan_src_real = torch.zeros(latent_t_fake.size(0)).float().cuda()
    #         gt_gan_trg_real = torch.ones(latent_trg.size(0)).float().cuda()
    #
    #     out_gan_real = torch.sigmoid(gan_model(latent_trg_src)).view(-1)
    #     gt_gan_real = torch.cat([gt_gan_src_real, gt_gan_trg_real], dim=0)
    #     loss_gan_real = gan_loss_fn(out_gan_real, gt_gan_real)
    #     loss_gan_real.backward()
    #     optimizer_gan.step()
    #
    #     g_model_lst[0].zero_grad()
    #     latent_trg = g_model_lst[0](image_t)
    #     out_gan_fake = torch.sigmoid(gan_model(latent_trg)).view(-1)
    #     gt_gan_fake = torch.zeros(latent_trg.size(0)).float().cuda()
    #     loss_gan_fake = gan_loss_fn(out_gan_fake, gt_gan_fake)
    #     loss_gan_fake.backward()
    #     optimizer_g_lst[0].step()
    mean_confident_gate = AverageMeter()
    mean_confident = AverageMeter()
    for i, (image_t, _) in enumerate(dloader_trg):
        if i > batch_per_epoch:
            break
        image_t = image_t.cuda()
        # label_t = label_t.long().cuda()

        ## TRAIN PHASE 2: Gt + Ht mixup
        optimizer_g_lst[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()

        with torch.no_grad():
            latent_trg = g_model_lst[1](image_t)
            output_dis_trg = dis_model(latent_trg)
            hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)

            hs_g_xt_conf, hs_g_xt_arg_max = hs_g_xt.max(1)
            hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)

        lam = np.random.beta(2, 2)
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]

        latent_trg_mixed = g_model_lst[0](mixed_image)
        ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
        # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)

        confidence_gate_i = torch.mean(hs_g_xt_conf.float()) * 1.2
        if epoch == 0:
            confidence_gate = confidence_gate_i.item()
        mean_confident_gate.update(confidence_gate_i.item())
        if confidence_gate < 0.3:
            confidence_gate = 0.3

        hs_g_xt_mask = (hs_g_xt_conf > confidence_gate).float().cuda()

        mean_confident.update(torch.mean(hs_g_xt_mask))
        loss_mimic_ht_g_xt_mixed = torch.mean(
            hs_g_xt_mask * torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))

        loss_mimic_ht_g_xt_mixed.backward()
        optimizer_g_lst[0].step()
        classifier_optimizer_list[0].step()

    domain_weight = [1 / (num_src_domain + 1)] * (num_src_domain + 1)
    # domain_weight.insert(0, domain_weight[-1] * 0.2)
    federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)

    # compute mmd:
    # latent_s = []
    # with torch.no_grad():
    #     for g_model, dataLoader in zip(g_model_lst[1:], train_dloader[1:]):
    #         latent_s_i = []
    #         for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
    #             if idx == 10:
    #                 break
    #             image_s_i = image_s_i.cuda()
    #             latent_s_i.append(g_model(image_s_i).view(batch_size, -1))
    #         latent_s.append(torch.cat(latent_s_i))
    # print("Compute MMD")
    # for i, s_n in enumerate(source_domains_name[:num_src_domain-1]):
    #     print(s_n)
    #     for j, s_n_j in enumerate(source_domains_name[i+1:]):
    #         print("\t{}: {}".format(s_n_j, compute_mmd(latent_s[i], latent_s[j])))

    #### MIX Generator\
    print("epoch {}:".format(epoch))
    for n_s, l_s in zip(source_domains_name, loss_class_src_lst):
        print("\tloss_class_{}: \t{:.4f}".format(n_s, l_s.item()))

    print("\tloss_hs_combine_g_xs: {:.4f}".format(loss_hs_combine.item()))
    print("MIMIC")
    print("\tloss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed.item()))
    print("\tmean_confident_hs_combine_g_xt: {:.4f}".format(mean_confident.avg))
    print("\tconfidence_gate: {:.4f}".format(confidence_gate))
    # acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
    # acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
    # acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
    # print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
    # print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
    # print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

    ###### print gan_loss_real
    # print("loss_gan_real: {:.4f}".format(loss_gan_real))

    if epoch == 0:
        confidence_gate = mean_confident_gate.avg
    else:
        confidence_gate = 0.8 * confidence_gate + 0.2 * mean_confident_gate.avg
    return confidence_gate

def train_multi_g_phase_4_separate_1_2_confi_1(train_dloader, g_model_lst, classifier_list, dis_model, gan_model,
                                       optimizer_g_lst,
                                       optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                       num_classes, source_domains_name, batchnorm_mmd, batch_per_epoch, confidence_gate):
    # source_domain_num = len(train_dloader_list[1:])
    print("updated")
    print(
        "train_multi_g_phase_4_separate_1_2_confi_1, mixup Ht, remove Gan, Gs_i, Gt, C and Hs_i train through hs_combine, update G_combine by Gs only, before train gt, update G_combine by Gs and Gt, after all, train only ht then")
    for g_model in g_model_lst:
        g_model.train()
    dis_model.train()
    # gan_model.train()
    for hs_i in classifier_list:
        hs_i.train()
    num_src_domain = 5

    loss_class_src_lst = []
    for g_model, opt_g, hs_i, opt_hs_i, dataLoader in zip(g_model_lst[1:], optimizer_g_lst[1:], classifier_list[1:],
                                                          classifier_optimizer_list[1:], train_dloader[1:]):
        for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
            if idx > batch_per_epoch:
                break
            image_s_i = image_s_i.cuda()
            label_s_i = label_s_i.long().cuda()

            # TRAIN PHASE 0
            opt_g.zero_grad()
            opt_hs_i.zero_grad()

            output_hs_i = hs_i(g_model(image_s_i))
            loss_class_src_i = softmaxcrossEntropy(output_hs_i, label_s_i)
            loss_class_src_lst.append(loss_class_src_i)

            loss_class_src_i.backward()
            opt_g.step()
            opt_hs_i.step()

    domain_weight = [1 / num_src_domain] * num_src_domain
    domain_weight.insert(0, 0)
    federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)

    i = -1
    dloader_trg = train_dloader[0]
    dloader_src_1 = train_dloader[1]
    dloader_src_2 = train_dloader[2]
    dloader_src_3 = train_dloader[3]
    dloader_src_4 = train_dloader[4]
    dloader_src_5 = train_dloader[5]
    for (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        i += 1
        if i == batch_per_epoch:
            break

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]
        optimizer_dis.zero_grad()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.zero_grad()
        # loss_hs_combine_lst = []
        latent_s = []

        with torch.no_grad():
            for idx in range(len(image_s_lst)):
                latent_s_i = g_model_lst[idx + 1](image_s_lst[idx])
                latent_s.append(latent_s_i)
            latent_s = torch.cat(latent_s)
        out_hs = []
        for classifier in classifier_list[1:]:
            out_hs_i = classifier(latent_s)
            out_hs.append(softmax_fn(out_hs_i))
        out_dis_s = dis_model(latent_s)
        output_hs_combine_g_xs = hs_gx_C_gx(out_hs, out_dis_s)
        label_s = torch.cat(label_s_lst)

        loss_hs_combine = crossEntropy_fn(output_hs_combine_g_xs, label_s, label_to_1hot=True)
        loss_hs_combine.backward()
        optimizer_dis.step()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.step()

    mean_confident_gate = AverageMeter()
    mean_confident = AverageMeter()
    for i, (image_t, _) in enumerate(dloader_trg):
        if i > batch_per_epoch:
            break
        image_t = image_t.cuda()
        # label_t = label_t.long().cuda()

        ## TRAIN PHASE 2: Gt + Ht mixup
        optimizer_g_lst[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()

        with torch.no_grad():
            latent_trg = g_model_lst[1](image_t)
            output_dis_trg = dis_model(latent_trg)
            hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)

            hs_g_xt_conf, hs_g_xt_arg_max = hs_g_xt.max(1)
            hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)

        lam = np.random.beta(2, 2)
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]

        latent_trg_mixed = g_model_lst[0](mixed_image)
        ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
        # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)

        confidence_gate_i = torch.mean(hs_g_xt_conf.float()) * 1.2
        if epoch == 0:
            confidence_gate = confidence_gate_i.item()
        mean_confident_gate.update(confidence_gate_i.item())
        if confidence_gate < 0.3:
            confidence_gate = 0.3

        hs_g_xt_mask = (hs_g_xt_conf > confidence_gate).float().cuda()

        mean_confident.update(torch.mean(hs_g_xt_mask))
        loss_mimic_ht_g_xt_mixed = torch.mean(
            hs_g_xt_mask * torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))

        loss_mimic_ht_g_xt_mixed.backward()
        optimizer_g_lst[0].step()
        classifier_optimizer_list[0].step()

    domain_weight = [1 / (num_src_domain + 1)] * (num_src_domain + 1)
    # domain_weight.insert(0, domain_weight[-1] * 0.2)
    federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)

    mean_confident_ht = AverageMeter()
    for i, (image_t, _) in enumerate(dloader_trg):
        if i > batch_per_epoch:
            break
        image_t = image_t.cuda()
        # label_t = label_t.long().cuda()

        ## TRAIN PHASE 2: Gt + Ht mixup
        # optimizer_g_lst[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()

        with torch.no_grad():
            latent_trg = g_model_lst[0](image_t)
            output_dis_trg = dis_model(latent_trg)
            hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)

            hs_g_xt_conf, hs_g_xt_arg_max = hs_g_xt.max(1)
            hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)

            lam = np.random.beta(2, 2)
            batch_size = image_t.size(0)
            index = torch.randperm(batch_size).cuda()
            mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
            mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]

            latent_trg_mixed = g_model_lst[0](mixed_image)
        ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
        # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)
        hs_g_xt_mask = (hs_g_xt_conf > confidence_gate).float().cuda()

        mean_confident_ht.update(torch.mean(hs_g_xt_mask))
        loss_mimic_ht_g_xt_mixed = torch.mean(
            hs_g_xt_mask * torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))

        loss_mimic_ht_g_xt_mixed.backward()
        # optimizer_g_lst[0].step()
        classifier_optimizer_list[0].step()


    print("epoch {}:".format(epoch))
    for n_s, l_s in zip(source_domains_name, loss_class_src_lst):
        print("\tloss_class_{}: \t{:.4f}".format(n_s, l_s.item()))

    print("\tloss_hs_combine_g_xs: {:.4f}".format(loss_hs_combine.item()))
    print("MIMIC")
    print("\tloss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed.item()))
    print("\tmean_confident_hs_combine_g_xt: {:.4f}".format(mean_confident.avg))
    print("\tmean_confident_hs_combine_g_xt_2: {:.4f}".format(mean_confident_ht.avg))
    print("\tconfidence_gate: {:.4f}".format(confidence_gate))

    if epoch == 0:
        confidence_gate = mean_confident_gate.avg
    else:
        confidence_gate = 0.8 * confidence_gate + 0.2 * mean_confident_gate.avg
    return confidence_gate


def train_multi_g_phase_4_separate_1_3(train_dloader, g_model_lst, classifier_list, dis_model, gan_model,
                                       optimizer_g_lst,
                                       optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                       num_classes, source_domains_name, batchnorm_mmd, batch_per_epoch):
    # source_domain_num = len(train_dloader_list[1:])
    print("updated")
    print(
        "train_multi_g_phase_4_separate_1_3, mixup Ht, remove Gan, Gs_i, Gt, C and Hs_i train through hs_combine with CONFIDENT, update G_combine by Gs only, before train gt, update G_combine by Gs and Gt, after all")
    for g_model in g_model_lst:
        g_model.train()
    dis_model.train()
    # gan_model.train()
    for hs_i in classifier_list:
        hs_i.train()
    num_src_domain = len(source_domains_name)

    loss_class_src_lst = []
    for g_model, opt_g, hs_i, opt_hs_i, dataLoader in zip(g_model_lst[1:], optimizer_g_lst[1:], classifier_list[1:],
                                                          classifier_optimizer_list[1:], train_dloader[1:]):
        for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
            if idx > batch_per_epoch:
                break
            image_s_i = image_s_i.cuda()
            label_s_i = label_s_i.long().cuda()

            # TRAIN PHASE 0
            opt_g.zero_grad()
            opt_hs_i.zero_grad()

            output_hs_i = hs_i(g_model(image_s_i))
            loss_class_src_i = softmaxcrossEntropy(output_hs_i, label_s_i)
            loss_class_src_lst.append(loss_class_src_i)

            loss_class_src_i.backward()
            opt_g.step()
            opt_hs_i.step()

    dis_conf = (0.85 - 0.5) * (epoch / 60) + 0.5
    if dis_conf > 0.85:
        dis_conf = 0.85

    i = -1
    dloader_trg = train_dloader[0]
    dloader_src_1 = train_dloader[1]
    dloader_src_2 = train_dloader[2]
    dloader_src_3 = train_dloader[3]
    dloader_src_4 = train_dloader[4]
    dloader_src_5 = train_dloader[5]
    for (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        i += 1
        if i == batch_per_epoch:
            break

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]
        optimizer_dis.zero_grad()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.zero_grad()
        # loss_hs_combine_lst = []
        latent_s = []

        with torch.no_grad():
            for idx in range(len(image_s_lst)):
                latent_s_i = g_model_lst[idx + 1](image_s_lst[idx])
                latent_s.append(latent_s_i)
            latent_s = torch.cat(latent_s)
        out_hs = []
        for classifier in classifier_list[1:]:
            out_hs_i = classifier(latent_s)
            out_hs.append(softmax_fn(out_hs_i))
        out_dis_s = dis_model(latent_s)
        output_hs_combine_g_xs = hs_gx_C_gx(out_hs, out_dis_s)
        label_s = torch.cat(label_s_lst)

        loss_hs_combine = crossEntropy_fn_withConf(output_hs_combine_g_xs, label_s, label_to_1hot=True,
                                                   confident_thr=dis_conf)
        loss_hs_combine.backward()
        optimizer_dis.step()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.step()

    ### PHASE 3: train Gan, Gs, Gt + GAN only #####
    # real_label_gan = 1.
    # fake_label_gan = 0.

    domain_weight = [1 / num_src_domain] * num_src_domain
    domain_weight.insert(0, 0)
    federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)

    mean_confident = AverageMeter()
    for i, (image_t, _) in enumerate(dloader_trg):
        if i > batch_per_epoch:
            break
        image_t = image_t.cuda()
        # label_t = label_t.long().cuda()

        ## TRAIN PHASE 2: Gt + Ht mixup
        optimizer_g_lst[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()

        with torch.no_grad():
            latent_trg = g_model_lst[1](image_t)
            output_dis_trg = dis_model(latent_trg)
            hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)
            hs_g_xt_max_by_class, hs_g_xt_arg_max = hs_g_xt.max(1)
            hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)

        lam = np.random.beta(2, 2)
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        # mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
        mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]

        latent_trg_mixed = g_model_lst[0](mixed_image)
        ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
        # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)

        confidence_gate = (0.95 - 0.3) * (epoch / 100) + 0.3
        if confidence_gate > 0.95:
            confidence_gate = 0.95
        hs_g_xt_mask = (hs_g_xt_max_by_class > confidence_gate).float().cuda()
        mean_confident.update(torch.mean(hs_g_xt_mask))
        loss_mimic_ht_g_xt_mixed = torch.mean(
            hs_g_xt_mask * torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
        # loss_mimic_ht_g_xt_mixed = torch.mean(
        #     torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))

        loss_mimic_ht_g_xt_mixed.backward()
        optimizer_g_lst[0].step()
        classifier_optimizer_list[0].step()

    # train time 2
    # domain_weight = [1 / (num_src_domain + 1)] * (num_src_domain + 1)
    # domain_weight.insert(0, domain_weight[-1] * 0.1)
    # federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)
    # train time 3
    domain_weight = [1 / (num_src_domain + 0.1)] * num_src_domain
    domain_weight.insert(0, domain_weight[-1] * 0.1)
    federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)

    # compute mmd:
    # latent_s = []
    # with torch.no_grad():
    #     for g_model, dataLoader in zip(g_model_lst[1:], train_dloader[1:]):
    #         latent_s_i = []
    #         for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
    #             if idx == 10:
    #                 break
    #             image_s_i = image_s_i.cuda()
    #             latent_s_i.append(g_model(image_s_i).view(batch_size, -1))
    #         latent_s.append(torch.cat(latent_s_i))
    # print("Compute MMD")
    # for i, s_n in enumerate(source_domains_name[:num_src_domain-1]):
    #     print(s_n)
    #     for j, s_n_j in enumerate(source_domains_name[i+1:]):
    #         print("\t{}: {}".format(s_n_j, compute_mmd(latent_s[i], latent_s[j])))

    #### MIX Generator\
    print("epoch {}:".format(epoch))
    for n_s, l_s in zip(source_domains_name, loss_class_src_lst):
        print("\tloss_class_{}: \t{:.4f}".format(n_s, l_s.item()))

    print("\tloss_hs_combine_g_xs: {:.4f}".format(loss_hs_combine.item()))
    print("MIMIC")
    print("\tloss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed.item()))
    print("\tmean_confident_hs_combine_g_xt: {:.4f}".format(mean_confident.avg))
    # acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
    # acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
    # acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
    # print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
    # print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
    # print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

    ###### print gan_loss_real
    # print("loss_gan_real: {:.4f}".format(loss_gan_real))


def train_multi_g_phase_4_separate_1_gan(train_dloader, g_model_lst, classifier_list, dis_model, gan_model,
                                         optimizer_g_lst,
                                         optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                         num_classes, source_domains, batchnorm_mmd, batch_per_epoch):
    # source_domain_num = len(train_dloader_list[1:])
    print("train_multi_g_phase_4_separate_1_GAN, mixup Ht, USING Gan, Gs_i, Gt, C and Hs_i train through hs_combine")
    for g_model in g_model_lst:
        g_model.train()
    dis_model.train()
    gan_model.train()
    for hs_i in classifier_list:
        hs_i.train()
    num_src_domain = 5

    dloader_trg = train_dloader[0]
    dloader_src_1 = train_dloader[1]
    dloader_src_2 = train_dloader[2]
    dloader_src_3 = train_dloader[3]
    dloader_src_4 = train_dloader[4]
    dloader_src_5 = train_dloader[5]
    ### PHASE 3: train Gan, Gs, Gt + GAN only #####
    # real_label_gan = 1.
    # fake_label_gan = 0.
    i = -1
    for (image_t, label_t), (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_trg, dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        i += 1
        if i > batch_per_epoch:
            break

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        # label_s_lst = [
        #     label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
        #     label_s_5.long().cuda()
        # ]
        image_t = image_t.cuda()
        gan_model.zero_grad()
        with torch.no_grad():
            latent_trg = g_model_lst[0](image_t)
            latent_s_lst = []
            for idx in range(len(image_s_lst)):
                latent_s_idx = g_model_lst[idx + 1](image_s_lst[idx])
                # latent_s_idx = g_model_lst[idx + 1](image_s_lst[idx][:4])
                latent_s_lst.append(latent_s_idx)

            latent_src = torch.cat(latent_s_lst, dim=0)
            latent_trg_src = torch.cat([latent_src, latent_trg], dim=0)
            gt_gan_src_real = torch.zeros(latent_src.size(0)).float().cuda()
            gt_gan_trg_real = torch.ones(latent_trg.size(0)).float().cuda()

        out_gan_real = torch.sigmoid(gan_model(latent_trg_src)).view(-1)
        gt_gan_real = torch.cat([gt_gan_src_real, gt_gan_trg_real], dim=0)
        loss_gan_real = gan_loss_fn(out_gan_real, gt_gan_real)
        loss_gan_real.backward()
        optimizer_gan.step()

        g_model_lst[0].zero_grad()
        latent_trg = g_model_lst[0](image_t)
        out_gan_fake = torch.sigmoid(gan_model(latent_trg)).view(-1)
        gt_gan_fake = torch.zeros(latent_trg.size(0)).float().cuda()
        loss_gan_fake = gan_loss_fn(out_gan_fake, gt_gan_fake)
        loss_gan_fake.backward()
        optimizer_g_lst[0].step()

    loss_class_src_lst = []
    for g_model, opt_g, hs_i, opt_hs_i, dataLoader in zip(g_model_lst[1:], optimizer_g_lst[1:], classifier_list[1:],
                                                          classifier_optimizer_list[1:], train_dloader[1:]):
        for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
            if idx > batch_per_epoch:
                break
            image_s_i = image_s_i.cuda()
            label_s_i = label_s_i.long().cuda()

            # TRAIN PHASE 0
            opt_g.zero_grad()
            opt_hs_i.zero_grad()

            output_hs_i = hs_i(g_model(image_s_i))
            loss_class_src_i = softmaxcrossEntropy(output_hs_i, label_s_i)
            loss_class_src_lst.append(loss_class_src_i)

            loss_class_src_i.backward()
            opt_g.step()
            opt_hs_i.step()

    i = -1
    for (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        i += 1
        if i == batch_per_epoch:
            break

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]
        optimizer_dis.zero_grad()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.zero_grad()
        # loss_hs_combine_lst = []
        latent_s = []

        with torch.no_grad():
            for idx in range(len(image_s_lst)):
                latent_s_i = g_model_lst[idx + 1](image_s_lst[idx])
                latent_s.append(latent_s_i)
            latent_s = torch.cat(latent_s)
        out_hs = []
        for classifier in classifier_list[1:]:
            out_hs_i = classifier(latent_s)
            out_hs.append(softmax_fn(out_hs_i))
        out_dis_s = dis_model(latent_s)
        output_hs_combine_g_xs = hs_gx_C_gx(out_hs, out_dis_s)
        label_s = torch.cat(label_s_lst)

        loss_hs_combine = crossEntropy_fn(output_hs_combine_g_xs, label_s, label_to_1hot=True)
        loss_hs_combine.backward()
        optimizer_dis.step()
        for opt_hs_i in classifier_optimizer_list[1:]:
            opt_hs_i.step()

    mean_confident = AverageMeter()
    for image_t, label_t in dloader_trg:
        image_t = image_t.cuda()
        # label_t = label_t.long().cuda()

        ## TRAIN PHASE 2: Gt + Ht mixup
        optimizer_g_lst[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()

        with torch.no_grad():
            latent_trg = g_model_lst[0](image_t)
            output_dis_trg = dis_model(latent_trg)
            hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)
            hs_g_xt_max_by_class, hs_g_xt_arg_max = hs_g_xt.max(1)
            hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)

        lam = np.random.beta(2, 2)
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        # mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
        mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]
        latent_trg_mixed = g_model_lst[0](mixed_image)
        ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
        # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)

        confidence_gate = (0.95 - 0.6) * (epoch / 60) + 0.6
        if confidence_gate > 0.95:
            confidence_gate = 0.95
        hs_g_xt_mask = (hs_g_xt_max_by_class > confidence_gate).float().cuda()
        mean_confident.update(torch.mean(hs_g_xt_mask))
        loss_mimic_ht_g_xt_mixed = torch.mean(
            hs_g_xt_mask * torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
        # loss_mimic_ht_g_xt_mixed = torch.mean(
        #     torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))

        loss_mimic_ht_g_xt_mixed.backward()
        optimizer_g_lst[0].step()
        classifier_optimizer_list[0].step()
    domain_weight = [1 / (num_src_domain + 1)] * (num_src_domain + 1)
    federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)

    #### MIX Generator\
    print("epoch {}:".format(epoch))
    print("gan_loss_real:\t{:.4f}".format(loss_gan_real.item()))
    print("gan_loss_fake:\t{:.4f}".format(loss_gan_fake.item()))
    for n_s, l_s in zip(source_domains, loss_class_src_lst):
        print("\tloss_class_{}: \t{:.4f}".format(n_s, l_s.item()))

    print("\tloss_hs_combine_g_xs: {:.4f}".format(loss_hs_combine.item()))
    print("MIMIC")
    print("\tloss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed.item()))
    print("\tmean_confident_hs_combine_g_xt: {:.4f}".format(mean_confident.avg))
    # acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
    # acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
    # acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
    # print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
    # print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
    # print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

    ###### print gan_loss_real
    # print("loss_gan_real: {:.4f}".format(loss_gan_real))


def train_multi_g_phase_4_separate(train_dloader, g_model_lst, classifier_list, dis_model, gan_model, optimizer_g_lst,
                                   optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                   num_classes, source_domains, batchnorm_mmd, batch_per_epoch):
    # source_domain_num = len(train_dloader_list[1:])
    print("train_multi_g_phase_4_separate, mixup Ht, remove Gan, Gs_i, Gt, C train through hs_combine")
    for g_model in g_model_lst:
        g_model.train()
    dis_model.train()
    # gan_model.train()
    for hs_i in classifier_list:
        hs_i.train()
    num_src_domain = 5

    dloader_trg = train_dloader[0]
    loss_class_src_lst = []
    for g_model, opt_g, hs_i, opt_hs_i, dataLoader in zip(g_model_lst[1:], optimizer_g_lst[1:], classifier_list[1:],
                                                          classifier_optimizer_list[1:], train_dloader[1:]):
        for idx, (image_s_i, label_s_i) in enumerate(dataLoader):
            if idx > batch_per_epoch:
                break
            image_s_i = image_s_i.cuda()
            label_s_i = label_s_i.long().cuda()

            # TRAIN PHASE 0
            opt_g.zero_grad()
            opt_hs_i.zero_grad()

            output_hs_i = hs_i(g_model(image_s_i))
            loss_class_src_i = softmaxcrossEntropy(output_hs_i, label_s_i)
            loss_class_src_lst.append(loss_class_src_i)

            loss_class_src_i.backward()
            opt_g.step()
            opt_hs_i.step()

    i = -1
    dloader_src_1 = train_dloader[1]
    dloader_src_2 = train_dloader[2]
    dloader_src_3 = train_dloader[3]
    dloader_src_4 = train_dloader[4]
    dloader_src_5 = train_dloader[5]
    for (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        i += 1
        if i == batch_per_epoch:
            break

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]
        optimizer_dis.zero_grad()
        loss_hs_combine_lst = []
        for idx in range(len(image_s_lst)):
            with torch.no_grad():
                latent_s_i = g_model_lst[idx + 1](image_s_lst[idx])
                out_hs = []
                for classifier in classifier_list[1:]:
                    out_hs_i = classifier(latent_s_i)
                    out_hs.append(softmax_fn(out_hs_i))
            out_dis_s_i = dis_model(latent_s_i)
            output_hs_combine_g_xs_i = hs_gx_C_gx(out_hs, out_dis_s_i)
            loss_hs_combine_lst.append(
                crossEntropy_fn(output_hs_combine_g_xs_i, label_s_lst[idx], label_to_1hot=True)
            )
        loss_hs_combine = sum(loss_hs_combine_lst)
        loss_hs_combine.backward()
        optimizer_dis.step()

    mean_confident = AverageMeter()
    for image_t, label_t in dloader_trg:
        image_t = image_t.cuda()
        # label_t = label_t.long().cuda()

        ## TRAIN PHASE 2: Gt + Ht mixup
        optimizer_g_lst[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()
        # th2: update C and Hs_i
        with torch.no_grad():
            latent_trg = g_model_lst[0](image_t)
            output_dis_trg = dis_model(latent_trg)
            hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)
            hs_g_xt_max_by_class, hs_g_xt_arg_max = hs_g_xt.max(1)
            hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)

        lam = np.random.beta(2, 2)
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        # mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
        mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]
        latent_trg_mixed = g_model_lst[0](mixed_image)
        ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
        # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)

        confidence_gate = (0.95 - 0.8) * (epoch / 60) + 0.8
        if confidence_gate > 0.95:
            confidence_gate = 0.95
        hs_g_xt_mask = (hs_g_xt_max_by_class > confidence_gate).float().cuda()
        mean_confident.update(torch.mean(hs_g_xt_mask))
        loss_mimic_ht_g_xt_mixed = torch.mean(
            hs_g_xt_mask * torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
        # loss_mimic_ht_g_xt_mixed = torch.mean(
        #     torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))

        loss_mimic_ht_g_xt_mixed.backward()
        optimizer_g_lst[0].step()
        classifier_optimizer_list[0].step()
    domain_weight = [1 / (num_src_domain + 1)] * (num_src_domain + 1)
    federated_average(g_model_lst, domain_weight, batchnorm_mmd=batchnorm_mmd)

    #### MIX Generator\
    print("epoch {}:".format(epoch))
    for n_s, l_s in zip(source_domains, loss_class_src_lst):
        print("\tloss_class_{}: \t{:.4f}".format(n_s, l_s.item()))

    print("\tloss_hs_combine_g_xs: {:.4f}".format(loss_hs_combine.item()))
    print("MIMIC")
    print("\tloss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed.item()))
    print("\tlmean_confident_hs_combine_g_xt: {:.4f}".format(mean_confident.avg))
    # acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
    # acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
    # acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
    # print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
    # print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
    # print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

    ###### print gan_loss_real
    # print("loss_gan_real: {:.4f}".format(loss_gan_real))


def train_multi_g_phase_3_separate_2(train_dloader, g_model_lst, classifier_list, dis_model, gan_model, optimizer_g_lst,
                                     optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                                     num_classes, source_domains, batchnorm_mmd, batch_per_epoch):
    # source_domain_num = len(train_dloader_list[1:])
    print("train_multi_g_phase_3_separate_2, mixup to C and Ht, remove Gan, Gs, Gt")
    gt_model = g_model_lst[0]
    gs_model = g_model_lst[1]

    gt_model.train()
    gs_model.train()
    dis_model.train()
    gan_model.train()
    for classifier in classifier_list:
        classifier.train()

    target_over_src = 5
    dloader_trg = train_dloader[0]
    dloader_src_1 = train_dloader[1]
    dloader_src_2 = train_dloader[2]
    dloader_src_3 = train_dloader[3]
    dloader_src_4 = train_dloader[4]
    dloader_src_5 = train_dloader[5]
    num_src_domain = 5
    i = -1
    for (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]

        # TRAIN PHASE 0
        optimizer_g_lst[1].zero_grad()
        optimizer_dis.zero_grad()
        for op_h in classifier_optimizer_list[1:]:
            op_h.zero_grad()

        # latent_trg = generator_model(image_t)
        latent_s_lst = []
        loss_class_src_lst = []

        gt_dis_src = []
        acc_src_hi_lst = []
        for idx in range(len(image_s_lst)):
            latent_s_idx = gs_model(image_s_lst[idx])
            output_hi_g_xi = classifier_list[idx + 1](latent_s_idx)
            task_loss_s = softmaxcrossEntropy(output_hi_g_xi, label_s_lst[idx])
            gt_dis_src_idx = torch.ones(output_hi_g_xi.size(0)) * idx
            gt_dis_src_idx = gt_dis_src_idx.long().cuda()
            # gt_dis_src_idx[:, idx] = 1
            gt_dis_src.append(gt_dis_src_idx)

            latent_s_lst.append(latent_s_idx)
            loss_class_src_lst.append(task_loss_s)
            acc_src_hi_lst.append(compute_acc(output_hi_g_xi, label_s_lst[idx]))

        loss_class_src_sum = sum(loss_class_src_lst)

        latent_src = torch.cat(latent_s_lst, dim=0)
        gt_dis_model = torch.cat(gt_dis_src, dim=0)

        output_dis_src = dis_model(latent_src)
        loss_dis_src = softmaxcrossEntropy(output_dis_src, gt_dis_model)
        # gt_gan_trg_fake = torch.zeros(latent_trg.size(0)).long().cuda()

        # out_gan_fake_trg = gan_model(latent_trg)
        # loss_gan_fake_trg = sigmoid_crossEntropy_withLogit_stable(out_gan_fake_trg, gt_gan_trg_fake)

        ### hs_combine_xs * c_xs
        output_hs_combine_g_xs = hs_combine_g_x(classifier_list, latent_src, output_dis_src, num_classes)
        gt_hs_combine_g_xs = torch.cat(label_s_lst)
        # print("gt_hs_combine_g_xs.size()", gt_hs_combine_g_xs.size())
        loss_mimic_hs_combine_g_xs = crossEntropy_fn(output_hs_combine_g_xs, gt_hs_combine_g_xs, label_to_1hot=True)

        # output_ht_g_xt = classifier_list[0](latent_trg)
        # output_ht_g_xs = classifier_list[0](latent_src)
        # loss_mimic_ht_g_xs = softmaxcrossEntropy(output_ht_g_xs, gt_hs_combine_g_xs)

        loss_phase0_lst = [loss_class_src_sum,
                           loss_dis_src,
                           # loss_ht_g_xt_entropy,
                           # loss_gan_fake_trg,
                           loss_mimic_hs_combine_g_xs,
                           # loss_mimic_hs_combine_g_xt,
                           # loss_mimic_ht_g_xt,
                           # loss_mimic_ht_g_xs,
                           ]

        loss_phase_0 = sum(loss_phase0_lst)
        loss_phase_0.backward()

        optimizer_g_lst[1].step()
        optimizer_dis.step()
        for op_h in classifier_optimizer_list[1:]:
            op_h.step()

        ## TRAIN PHASE 1: G + C + mixup
        optimizer_g_lst[1].zero_grad()
        optimizer_dis.zero_grad()

        gt_dis_src = []

        for idx in range(len(image_s_lst)):
            gt_dis_src_idx = torch.ones(image_s_lst[idx].size(0)) * idx
            gt_dis_src_idx = gt_dis_src_idx.long().cuda()
            gt_dis_src.append(gt_dis_src_idx)
        gt_dis_model = torch.cat(gt_dis_src)
        gt_dis_model_1hot = label_to_1hot(gt_dis_model, len(image_s_lst))

        image_s = torch.cat(image_s_lst)
        lam = np.random.beta(2, 2)
        batch_size = image_s.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image_s = lam * image_s + (1 - lam) * image_s[index, :]
        mixed_gt_dis_model = lam * gt_dis_model_1hot + (1 - lam) * gt_dis_model_1hot[index, :]

        latent_s_lst = []
        minibatch = batch_size // len(image_s_lst)
        for idx in range(len(image_s_lst)):
            latent_s_idx = gs_model(mixed_image_s[idx * minibatch: (idx + 1) * minibatch])
            latent_s_lst.append(latent_s_idx)
        latent_src = torch.cat(latent_s_lst)
        output_dis_src_mixed = dis_model(latent_src)
        loss_dis_src_mixed = softmaxCrossEntropy_withLogit(output_dis_src_mixed, mixed_gt_dis_model)

        loss_dis_src_mixed.backward()
        optimizer_g_lst[1].step()
        optimizer_dis.step()

        ### PHASE 3: train Gan only #####

        # optimizer_gan.zero_grad()
        # with torch.no_grad():
        #     latent_trg = generator_model(image_t)
        #     latent_s_lst = []
        #     for idx in range(len(image_s_lst)):
        #         latent_s_idx = generator_model(image_s_lst[idx])
        #         latent_s_lst.append(latent_s_idx)
        #
        #     latent_src = torch.cat(latent_s_lst, dim=0)
        #     latent_trg_src = torch.cat([latent_src, latent_trg], dim=0)
        # gt_gan_src_real = torch.zeros(latent_src.size(0)).long().cuda()
        # gt_gan_trg_real = torch.ones(latent_trg.size(0)).long().cuda()
        #
        # out_gan_real = gan_model(latent_trg_src)
        # gt_gan_real = torch.cat([gt_gan_src_real, gt_gan_trg_real], dim=0)
        # loss_gan_real = sigmoid_crossEntropy_withLogit_stable(out_gan_real, gt_gan_real)
        # loss_gan_real.backward()
        # optimizer_gan.step()

        # if i == batch_per_epoch - 1 or i == 0:
    for image_t, label_t in dloader_trg:
        image_t = image_t.cuda()
        # label_t = label_t.long().cuda()

        ## TRAIN PHASE 2: Gt + Ht mixup
        optimizer_g_lst[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()

        with torch.no_grad():
            latent_trg = gs_model(image_t)
            output_dis_trg = dis_model(latent_trg)
            hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)
            hs_g_xt_max_by_class, hs_g_xt_arg_max = hs_g_xt.max(1)
            hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)

        lam = np.random.beta(2, 2)
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        # mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
        mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]
        latent_trg_mixed = gt_model(mixed_image)
        ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
        # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)

        confidence_gate = (0.95 - 0.8) * (epoch / 60) + 0.8
        if confidence_gate > 0.95:
            confidence_gate = 0.95
        hs_g_xt_mask = (hs_g_xt_max_by_class > confidence_gate).float().cuda()
        loss_mimic_ht_g_xt_mixed = torch.mean(
            hs_g_xt_mask * torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
        # loss_mimic_ht_g_xt_mixed = torch.mean(
        #     torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))

        loss_mimic_ht_g_xt_mixed.backward()
        optimizer_g_lst[0].step()
        classifier_optimizer_list[0].step()

    ### PHASE 3: train Gan only #####

    # optimizer_gan.zero_grad()
    # with torch.no_grad():
    #     latent_trg = generator_model(image_t)
    #     latent_s_lst = []
    #     for idx in range(len(image_s_lst)):
    #         latent_s_idx = generator_model(image_s_lst[idx])
    #         latent_s_lst.append(latent_s_idx)
    #
    #     latent_src = torch.cat(latent_s_lst, dim=0)
    #     latent_trg_src = torch.cat([latent_src, latent_trg], dim=0)
    # gt_gan_src_real = torch.zeros(latent_src.size(0)).long().cuda()
    # gt_gan_trg_real = torch.ones(latent_trg.size(0)).long().cuda()
    #
    # out_gan_real = gan_model(latent_trg_src)
    # gt_gan_real = torch.cat([gt_gan_src_real, gt_gan_trg_real], dim=0)
    # loss_gan_real = sigmoid_crossEntropy_withLogit_stable(out_gan_real, gt_gan_real)
    # loss_gan_real.backward()
    # optimizer_gan.step()

    # if i == batch_per_epoch - 1 or i == 0:
    loss_name_lst = ["loss_class_src_sum",
                     "loss_dis_src",
                     # "loss_ht_g_xt_entropy",
                     # "loss_gan_fake_trg",
                     "loss_mimic_hs_combine_g_xs",
                     # "loss_mimic_hs_combine_g_xt",
                     # "loss_mimic_ht_g_xt",
                     # "loss_mimic_ht_g_xs"
                     ]
    print("epoch {}:\n\tloss_phase_0:\t{:.4f}".format(epoch, loss_phase_0.item()))
    for _n, _l in zip(loss_name_lst, loss_phase0_lst):
        print("{}: \t{:.4f}".format(_n, _l.item()))

    for idx, acc_s in enumerate(acc_src_hi_lst):
        print(
            "\t{}: loss: {:.4f}\tacc: {}".format(source_domains[idx], loss_class_src_lst[idx].item(), acc_s * 100.0))
    print("loss_dis_src_mixed: {:.4f}".format(loss_dis_src_mixed.item()))
    print("MIMIC")
    print("loss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed.item()))
    # acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
    # acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
    # acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
    # print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
    # print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
    # print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

    ###### print gan_loss_real
    # print("loss_gan_real: {:.4f}".format(loss_gan_real))


def test_trg(target_domain, source_domains, test_dloader_list, g_model_lst, dis_model, gan_model, classifier_list,
             epoch, writer,
             num_classes=126, top_5_accuracy=True, test_iter=None, name_summary="Test"):
    source_domain_losses = [AverageMeter() for i in source_domains]
    loss_ht_g_xt = AverageMeter()
    loss_hs_g_xt = AverageMeter()
    acc_ht_g_xt = AverageMeter()
    acc_hs_g_xt = AverageMeter()

    for g_model in g_model_lst:
        g_model.eval()
    dis_model.eval()
    # gan_model.eval()
    for classifier in classifier_list:
        classifier.eval()
    # calculate loss, accuracy for target domain
    # tmp_score = []
    # tmp_label = []
    output_dis_model_t_lst = []
    test_dloader_t = test_dloader_list[0]
    for idx, (image_t, label_t) in enumerate(test_dloader_t):
        if test_iter is not None and idx >= test_iter:
            break
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()
        with torch.no_grad():
            gt_xt = g_model_lst[0](image_t)
            output_ht_g_xt = classifier_list[0](gt_xt)

            # gs_xt = g_model_lst[1](image_t)
            output_dis_model_t = dis_model(gt_xt)
            output_hs_combine_g_xt = hs_combine_g_x(classifier_list, gt_xt, output_dis_model_t, num_classes)

        output_dis_model_t_lst.append(output_dis_model_t)
        task_loss_t = softmaxcrossEntropy(output_ht_g_xt, label_t)
        teacher_loss_t = softmaxcrossEntropy(output_hs_combine_g_xt, label_t)
        loss_ht_g_xt.update(float(task_loss_t.item()), image_t.size(0))
        loss_hs_g_xt.update(float(teacher_loss_t.item()), image_t.size(0))

        acc_ht_g_xt.update(float(compute_acc(output_ht_g_xt, label_t)), image_t.size(0))
        acc_hs_g_xt.update(float(compute_acc(output_hs_combine_g_xt, label_t)), image_t.size(0))

    writer.add_scalar(tag="{}/ht_g_xt_{}_loss".format(name_summary, target_domain), scalar_value=loss_ht_g_xt.avg,
                      global_step=epoch + 1)
    writer.add_scalar(tag="{}/hs_g_xt_{}_loss".format(name_summary, target_domain), scalar_value=loss_hs_g_xt.avg,
                      global_step=epoch + 1)

    writer.add_scalar(tag="{}/acc_ht_g_xt_{}".format(name_summary, target_domain).format(target_domain),
                      scalar_value=acc_ht_g_xt.avg,
                      global_step=epoch + 1)
    writer.add_scalar(tag="{}/acc_hs_g_xt_{}".format(name_summary, target_domain).format(target_domain),
                      scalar_value=acc_hs_g_xt.avg,
                      global_step=epoch + 1)
    print("Target Domain {}:\n\tacc_ht_g_xt {:.3f}\n\tacc_hs_g_xt {:.3f}".format(target_domain, acc_ht_g_xt.avg * 100.0,
                                                                                 acc_hs_g_xt.avg * 100.0))
    best_acc_i = max(acc_ht_g_xt.avg, acc_hs_g_xt.avg)
    output_dis_model_t_lst = torch.mean(softmax_fn(torch.cat(output_dis_model_t_lst)), dim=0)
    print("output_dis_model_t_mean", output_dis_model_t_lst)

    # # calculate loss, accuracy for source domains
    for s_i, domain_s in enumerate(source_domains):
        # tmp_score = []
        # tmp_label = []
        acc_hs_g_xs = AverageMeter()
        acc_dis = AverageMeter()
        acc_fn = AverageMeter()
        test_dloader_s = test_dloader_list[s_i + 1]
        for it, (image_s, label_s) in enumerate(test_dloader_s):
            if test_iter is not None and it >= test_iter:
                break
            image_s = image_s.cuda()
            label_s = label_s.long().cuda()
            with torch.no_grad():
                gs_xs = g_model_lst[s_i + 1](image_s)
                output_s = classifier_list[s_i + 1](gs_xs)
                out_dis_model_s = dis_model(gs_xs)
                gt_dis_src_idx = torch.ones(out_dis_model_s.size(0)) * s_i
                gt_dis_src_idx = gt_dis_src_idx.long().cuda()
                output_hs_combine = hs_combine_g_x(classifier_list, gs_xs, out_dis_model_s, num_classes)
            # label_onehot_s = torch.zeros(label_s.size(0), num_classes).cuda().scatter_(1, label_s.view(-1, 1), 1)
            task_loss_s = softmaxcrossEntropy(output_s, label_s)
            source_domain_losses[s_i].update(float(task_loss_s.item()), image_s.size(0))
            # tmp_score.append(torch.softmax(output_s, dim=1))
            acc_hs_g_xs.update(float(compute_acc(output_hs_combine, label_s)), image_s.size(0))
            acc_dis.update(float(compute_acc(out_dis_model_s, gt_dis_src_idx)), image_s.size(0))
            acc_fn.update(float(compute_acc(output_s, label_s)), image_s.size(0))
            # turn label into one-hot code
            # tmp_label.append(label_onehot_s)
        writer.add_scalar(tag="{}/source_domain_{}_loss".format(name_summary, domain_s),
                          scalar_value=source_domain_losses[s_i].avg,
                          global_step=epoch + 1)
        # tmp_score = torch.cat(tmp_score, dim=0).detach()
        # tmp_label = torch.cat(tmp_label, dim=0).detach()
        # _, y_true = torch.topk(tmp_label, k=1, dim=1)
        # _, y_pred = torch.topk(tmp_score, k=5, dim=1)
        # top_1_accuracy_s = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
        writer.add_scalar(tag="{}/src_{}_acc_hi".format(name_summary, domain_s),
                          scalar_value=acc_fn.avg * 100.0,
                          global_step=epoch + 1)
        writer.add_scalar(tag="{}/src_{}_acc_hs_combine".format(name_summary, domain_s),
                          scalar_value=acc_hs_g_xs.avg * 100.0,
                          global_step=epoch + 1)
        writer.add_scalar(tag="{}/src_{}_acc_dis".format(name_summary, domain_s),
                          scalar_value=acc_dis.avg * 100.0,
                          global_step=epoch + 1)
        print("\t{} \tloss: {:.3f}\tacc: {:.3f}\tacc_hs_g_xs: {:.3f}\tacc_dis: {:.3f}".format(domain_s,
                                                                                              source_domain_losses[
                                                                                                  s_i].avg,
                                                                                              acc_fn.avg * 100.0,
                                                                                              acc_hs_g_xs.avg * 100.0,
                                                                                              acc_dis.avg * 100.0))

    return best_acc_i