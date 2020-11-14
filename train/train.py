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
    prob = torch.clamp(prob, 1e-9, 1 - 1e-9)
    if label_to_1hot:
        num_classes = prob.size(1)
        label = torch.zeros(label.size(0), num_classes).cuda().scatter_(1, label.view(-1, 1), 1)
    return torch.mean(torch.sum(-1.0 * label * torch.log(prob), dim=-1))


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


def hs_combine_g_x(classifier_list, latent_v, out_dis_model, num_class):
    out_hs_sofmax_lst = []
    for classifier in classifier_list[1:]:
        out_hs_sofmax_lst.append(softmax_fn(classifier(latent_v)))

    out_hs_softmax = torch.stack(out_hs_sofmax_lst)  # (num_domain, batch_size, num_class)
    out_hs_softmax = torch.transpose(out_hs_softmax, 0, 1)  # (batch_size, num_domain, num_class)

    out_dis_model_softmax = softmax_fn(out_dis_model.detach().clone())  # (batch_size, num_domain, 1)
    out_dis_model_expand = out_dis_model_softmax.unsqueeze(-1)  # (batch_size, num_domain, 1)
    out_dis_model_repeat = out_dis_model_expand.repeat((1, 1, num_class))  # (batch_size, num_domain, num_class)

    hs_combine_c_g_x = out_hs_softmax * out_dis_model_repeat  # (batch_size, num_domain, num_class)
    hs_combine_c_g_x = torch.sum(hs_combine_c_g_x, dim=1)  # (batch_size, num_class)

    return hs_combine_c_g_x

def hs_gx_C_gx(out_hs_sofmax_lst, out_dis_model, num_class):
    out_hs_softmax = torch.stack(out_hs_sofmax_lst)  # (num_domain, batch_size, num_class) y^_i
    out_hs_softmax = torch.transpose(out_hs_softmax, 0, 1)  # (batch_size, num_domain, num_class)

    out_dis_model_softmax = softmax_fn(out_dis_model)  # (batch_size, num_domain)
    out_dis_model_expand = out_dis_model_softmax.unsqueeze(-1)  # (batch_size, num_domain, 1)
    out_dis_model_repeat = out_dis_model_expand.repeat((1, 1, out_hs_softmax.size(2)))  # (batch_size, num_domain, num_class)

    hs_combine_c_g_x = out_hs_softmax * out_dis_model_repeat  # (batch_size, num_domain, num_class)
    hs_combine_c_g_x = torch.sum(hs_combine_c_g_x, dim=1)  # (batch_size, num_class)

    return hs_combine_c_g_x

def train_model(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                  optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                  num_classes, source_domains, batchnorm_mmd, batch_per_epoch, training_mode=2):
    if training_mode == "1":
        train_phase_1(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                  optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                  num_classes, source_domains, batchnorm_mmd, batch_per_epoch)
    elif training_mode == "2":
        train_phase_2(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                      optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                      num_classes, source_domains, batchnorm_mmd, batch_per_epoch)
    elif training_mode == "3":
        train_phase_3(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                      optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                      num_classes, source_domains, batchnorm_mmd, batch_per_epoch)
    elif training_mode == "3_sep":
        train_phase_3_separate(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                      optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                      num_classes, source_domains, batchnorm_mmd, batch_per_epoch)
    elif training_mode == "3_sep_0":
        train_phase_3_separate_0(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                      optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                      num_classes, source_domains, batchnorm_mmd, batch_per_epoch)
    elif training_mode == "3_sep_1":
        train_phase_3_separate_1(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                      optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                      num_classes, source_domains, batchnorm_mmd, batch_per_epoch)
    elif training_mode == "3_sep_2":
        train_phase_3_separate_2(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                      optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                      num_classes, source_domains, batchnorm_mmd, batch_per_epoch)
    elif training_mode == "4_sep":
        train_phase_4_separate(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                      optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                      num_classes, source_domains, batchnorm_mmd, batch_per_epoch)
    elif training_mode == 4:
        train_phase_4(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                      optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                      num_classes, source_domains, batchnorm_mmd, batch_per_epoch)
    else:
        print("Cannot find", training_mode)


def train(train_dloader_list, generator_model, classifier_list, dis_model, gan_model, optimizer_g, optimizer_dis,
          optimizer_gan,
          classifier_optimizer_list, epoch, writer,
          num_classes, domain_weight, source_domains, batchnorm_mmd, batch_per_epoch, confidence_gate_begin,
          confidence_gate_end, communication_rounds, total_epochs, malicious_domain, attack_level):
    source_domain_num = len(train_dloader_list[1:])
    generator_model.train()
    for classifier in classifier_list:
        classifier.train()
    # If communication rounds <1,
    # then we perform parameter aggregation after (1/communication_rounds) epochs
    # If communication rounds >=1:
    # then we extend the training epochs and use fewer samples in each epoch.
    if communication_rounds in [0.2, 0.5]:
        model_aggregation_frequency = round(1 / communication_rounds)
    else:
        model_aggregation_frequency = 1

    src_train_dloader = train_dloader_list[1]
    for f in range(model_aggregation_frequency):
        # current_domain_index = 0
        # Train model locally on source domains
        for i, (image_s_lst, label_s_lst) in enumerate(src_train_dloader):
            if i == batch_per_epoch:
                break

            optimizer_g.zero_grad()
            for op_h in classifier_optimizer_list[1:]:
                op_h.zero_grad()
            loss_class_src_lst = []
            image_s_lst = image_s_lst.cuda()
            label_s_lst = label_s_lst.long().cuda()
            for idx, classifier in enumerate(classifier_list):
                if idx == 0:
                    continue
                image_s = image_s_lst[:, idx - 1]
                label_s = label_s_lst[:, idx - 1]
                # each source domain do optimize
                feature_s = generator_model(image_s)
                output_s = classifier(feature_s)
                task_loss_s = softmaxcrossEntropy(output_s, label_s)
                loss_class_src_lst.append(task_loss_s)

            loss_class_src_sum = loss_phase_1(loss_class_src_lst)
            if i == batch_per_epoch - 1 or i == 0:
                print(loss_class_src_sum.item())
            loss_class_src_sum.backward(retain_graph=True)
            optimizer_g.step()

            for op_h in classifier_optimizer_list[1:]:
                op_h.step()
    # # Domain adaptation on target domain
    # confidence_gate = (confidence_gate_end - confidence_gate_begin) * (epoch / total_epochs) + confidence_gate_begin
    # # We use I(n_i>=1)/(N_T) to adjust the weight for knowledge distillation domain
    # target_weight = [0, 0]
    # consensus_focus_dict = {}
    # for i in range(1, len(train_dloader_list)):
    #     consensus_focus_dict[i] = 0
    # for i, (image_t, label_t) in enumerate(train_dloader_list[0]):
    #     if i >= batch_per_epoch:
    #         break
    #     optimizer_list[0].zero_grad()
    #     classifier_optimizer_list[0].zero_grad()
    #     image_t = image_t.cuda()
    #     # Knowledge Vote
    #     with torch.no_grad():
    #         knowledge_list = [torch.softmax(classifier_list[i](generater_model_list[i](image_t)), dim=1).unsqueeze(1) for
    #                           i in range(1, source_domain_num + 1)]
    #         knowledge_list = torch.cat(knowledge_list, 1)
    #     _, consensus_knowledge, consensus_weight = knowledge_vote(knowledge_list, confidence_gate,
    #                                                               num_classes=num_classes)
    #     target_weight[0] += torch.sum(consensus_weight).item()
    #     target_weight[1] += consensus_weight.size(0)
    #     # Perform data augmentation with mixup
    #     lam = np.random.beta(2, 2)
    #     batch_size = image_t.size(0)
    #     index = torch.randperm(batch_size).cuda()
    #     mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
    #     mixed_consensus = lam * consensus_knowledge + (1 - lam) * consensus_knowledge[index, :]
    #     feature_t = generater_model_list[0](mixed_image)
    #     output_t = classifier_list[0](feature_t)
    #     output_t = torch.log_softmax(output_t, dim=1)
    #     task_loss_t = torch.mean(consensus_weight * torch.sum(-1 * mixed_consensus * output_t, dim=1))
    #     task_loss_t.backward()
    #     optimizer_list[0].step()
    #     classifier_optimizer_list[0].step()
    #     # Calculate consensus focus
    #     consensus_focus_dict = calculate_consensus_focus(consensus_focus_dict, knowledge_list, confidence_gate,
    #                                                      source_domain_num, num_classes)
    # # Consensus Focus Re-weighting
    # target_parameter_alpha = target_weight[0] / target_weight[1]
    # target_weight = round(target_parameter_alpha / (source_domain_num + 1), 4)
    # epoch_domain_weight = []
    # source_total_weight = 1 - target_weight
    # for i in range(1, source_domain_num + 1):
    #     epoch_domain_weight.append(consensus_focus_dict[i])
    # if sum(epoch_domain_weight) == 0:
    #     epoch_domain_weight = [v + 1e-3 for v in epoch_domain_weight]
    # epoch_domain_weight = [round(source_total_weight * v / sum(epoch_domain_weight), 4) for v in
    #                        epoch_domain_weight]
    # epoch_domain_weight.insert(0, target_weight)
    # # Update domain weight with moving average
    # if epoch == 0:
    #     domain_weight = epoch_domain_weight
    # else:
    #     domain_weight = update_domain_weight(domain_weight, epoch_domain_weight)
    # # Model aggregation and Batchnorm MMD
    # federated_average(generater_model_list, domain_weight, batchnorm_mmd=batchnorm_mmd)
    # # Recording domain weight in logs
    # writer.add_scalar(tag="Train/target_domain_weight", scalar_value=target_weight, global_step=epoch + 1)
    # for i in range(0, len(train_dloader_list) - 1):
    #     writer.add_scalar(tag="Train/source_domain_{}_weight".format(source_domains[i]),
    #                       scalar_value=domain_weight[i + 1], global_step=epoch + 1)
    # print("Source Domains:{}, Domain Weight :{}".format(source_domains, domain_weight[1:]))
    # return domain_weight


def train_phase_1(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                  optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                  num_classes, source_domains, batchnorm_mmd, batch_per_epoch):
    # source_domain_num = len(train_dloader_list[1:])
    generator_model.train()
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
    for (image_t, label_t), (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_trg, dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        optimizer_g.zero_grad()
        optimizer_dis.zero_grad()
        for op_h in classifier_optimizer_list:
            op_h.zero_grad()

        i += 1
        if i == batch_per_epoch:
            break
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]

        latent_trg = generator_model(image_t)
        latent_s_lst = []
        loss_class_src_lst = []

        gt_dis_src = []
        acc_src_hi_lst = []
        for idx in range(len(image_s_lst)):
            latent_s_idx = generator_model(image_s_lst[idx])
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

        output_dis_trg = dis_model(latent_trg)
        loss_dis_trg_entropy = cent(output_dis_trg)
        # latent_trg_src = torch.cat([latent_src, latent_trg], dim=0)

        # gt_gan_src_real = torch.zeros(latent_src.size(0)).long().cuda()
        # gt_gan_src_fake = torch.ones(latent_src.size(0)).long().cuda()
        #
        # gt_gan_trg_real = torch.ones(latent_trg.size(0)).long().cuda()
        # gt_gan_trg_fake = torch.zeros(latent_trg.size(0)).long().cuda()
        #
        # gt_gan_fake = torch.cat((gt_gan_src_fake, gt_gan_trg_real), dim=0)
        # out_gan_src_trg = gan_model(latent_trg_src)
        # loss_gan_fake = softmaxcrossEntropy(out_gan_src_trg, gt_gan_fake)
        #
        # gt_gan_real = torch.cat((gt_gan_src_real, gt_gan_trg_fake), dim=0)
        # print(gt_gan_real.size())
        # loss_gan_real = softmaxcrossEntropy(out_gan_src_trg, gt_gan_real)

        ### hs_combine_xs * c_xs

        output_hs_combine_g_xs = hs_combine_g_x(classifier_list, latent_src, output_dis_src, num_classes)
        gt_hs_combine_g_xs = torch.cat(label_s_lst)
        # print("gt_hs_combine_g_xs.size()", gt_hs_combine_g_xs.size())
        loss_mimic_hs_combine_g_xs = crossEntropy_fn(output_hs_combine_g_xs, gt_hs_combine_g_xs, label_to_1hot=True)
        # print(loss_mimic_hs_combine_g_xs.item())
        ### hs_combine_xt * c_xt
        # with torch.no_grad():
        output_hs_combine_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)

        # loss_mimic_hs_combine_g_xt = crossEntropy_fn(output_hs_combine_g_xt, output_hs_combine_g_xt_argmax,
        #                                              label_to_1hot=True)

        output_ht_g_xt = classifier_list[0](latent_trg)
        loss_ht_g_xt_entropy = cent(output_ht_g_xt)
        # output_hs_combine_g_xt_1hot = label_to_1hot(output_hs_combine_g_xt_argmax, num_classes)
        # loss_mimic_ht_g_xt = crossEntropy_fn(softmax_fn(output_ht_g_xt), output_hs_combine_g_xt)
        # loss_mimic_ht_g_xt = softmaxCrossEntropy_withLogit(output_ht_g_xt, output_hs_combine_g_xt)

        for param in generator_model.parameters():
            param.requires_grad = True

        loss_pahse2_lst = [loss_class_src_sum,
                           loss_dis_src,
                           loss_ht_g_xt_entropy,
                           # loss_gan_fake,
                           loss_mimic_hs_combine_g_xs,
                           # loss_mimic_hs_combine_g_xt,
                           # loss_mimic_ht_g_xt
                           ]

        loss_phase_2 = sum(loss_pahse2_lst)
        loss_phase_2.backward()
        optimizer_g.step()
        optimizer_dis.step()
        for op_h in classifier_optimizer_list:
            op_h.step()

        ### train Gan only #####
        # for param in generator_model.parameters():
        #     param.requires_grad = False
        #
        # optimizer_gan.zero_grad()
        # loss_gan_real.backward()
        # optimizer_gan.step()

        if i == batch_per_epoch - 1 or i == 0:
            loss_pahse2_name_lst = ["loss_class_src_sum",
                               "loss_dis_src",
                               "loss_ht_g_xt_entropy",
                               # "loss_gan_fake",
                               "loss_mimic_hs_combine_g_xs",
                               # "loss_mimic_hs_combine_g_xt",
                               # "loss_mimic_ht_g_xt"
                                    ]
            print("epoch {}:\n\tloss_phase_2:\t{}".format(epoch, loss_phase_2.item()))
            for _n, _l in zip(loss_pahse2_name_lst, loss_pahse2_lst):
                print("{}: \t{}".format(_n, _l.item()))

            for idx, acc_s in enumerate(acc_src_hi_lst):
                print(
                    "{}: loss: {}\tacc: {}".format(source_domains[idx], loss_class_src_lst[idx].item(), acc_s * 100.0))

            print("MIMIC")
            acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
            acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
            acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
            print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
            print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
            print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

def train_phase_2(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                  optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                  num_classes, source_domains, batchnorm_mmd, batch_per_epoch):
    # source_domain_num = len(train_dloader_list[1:])
    generator_model.train()
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
    for (image_t, label_t), (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_trg, dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        optimizer_g.zero_grad()
        optimizer_dis.zero_grad()
        for op_h in classifier_optimizer_list:
            op_h.zero_grad()

        i += 1
        if i == batch_per_epoch:
            break
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]

        latent_trg = generator_model(image_t)
        latent_s_lst = []
        loss_class_src_lst = []

        gt_dis_src = []
        acc_src_hi_lst = []
        for idx in range(len(image_s_lst)):
            latent_s_idx = generator_model(image_s_lst[idx])
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

        output_dis_trg = dis_model(latent_trg)
        loss_dis_trg_entropy = cent(output_dis_trg)

        latent_trg_src = torch.cat([latent_src, latent_trg], dim=0)

        gt_gan_src_real = torch.zeros(latent_src.size(0)).long().cuda()
        gt_gan_src_fake = torch.ones(latent_src.size(0)).long().cuda()

        gt_gan_trg_real = torch.ones(latent_trg.size(0)).long().cuda()
        gt_gan_trg_fake = torch.zeros(latent_trg.size(0)).long().cuda()

        out_gan_fake = gan_model(latent_trg_src)

        gt_gan_fake = torch.cat((gt_gan_src_fake, gt_gan_trg_fake), dim=0)
        loss_gan_fake = sigmoid_crossEntropy_withLogit_stable(out_gan_fake, gt_gan_fake)

        ### hs_combine_xs * c_xs
        output_hs_combine_g_xs = hs_combine_g_x(classifier_list, latent_src, output_dis_src, num_classes)
        gt_hs_combine_g_xs = torch.cat(label_s_lst)
        # print("gt_hs_combine_g_xs.size()", gt_hs_combine_g_xs.size())
        loss_mimic_hs_combine_g_xs = crossEntropy_fn(output_hs_combine_g_xs, gt_hs_combine_g_xs, label_to_1hot=True)
        # print(loss_mimic_hs_combine_g_xs.item())
        ### hs_combine_xt * c_xt
        # with torch.no_grad():
        output_hs_combine_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)

        # loss_mimic_hs_combine_g_xt = crossEntropy_fn(output_hs_combine_g_xt, output_hs_combine_g_xt_argmax,
        #                                              label_to_1hot=True)

        output_ht_g_xt = classifier_list[0](latent_trg)
        loss_ht_g_xt_entropy = cent(output_ht_g_xt)
        # output_hs_combine_g_xt_1hot = label_to_1hot(output_hs_combine_g_xt_argmax, num_classes)
        # loss_mimic_ht_g_xt = crossEntropy_fn(softmax_fn(output_ht_g_xt), output_hs_combine_g_xt)
        loss_mimic_ht_g_xt = softmaxCrossEntropy_withLogit(output_ht_g_xt, output_hs_combine_g_xt)

        # for param in generator_model.parameters():
        #     param.requires_grad = True

        loss_lst = [loss_class_src_sum,
                           loss_dis_src,
                           loss_ht_g_xt_entropy,
                           loss_gan_fake,
                           loss_mimic_hs_combine_g_xs,
                           # loss_mimic_hs_combine_g_xt,
                           # loss_mimic_ht_g_xt
                           ]

        loss_phase_2 = sum(loss_lst)
        loss_phase_2.backward()
        optimizer_g.step()
        optimizer_dis.step()
        for op_h in classifier_optimizer_list:
            op_h.step()

        ### train Gan only #####
        latent_trg = generator_model(image_t)
        latent_s_lst = []
        for idx in range(len(image_s_lst)):
            latent_s_idx = generator_model(image_s_lst[idx])
            latent_s_lst.append(latent_s_idx)

        latent_src = torch.cat(latent_s_lst, dim=0)
        latent_trg_src = torch.cat([latent_src, latent_trg], dim=0)
        out_gan_real = gan_model(latent_trg_src)
        gt_gan_real = torch.cat([gt_gan_src_real, gt_gan_trg_real], dim=0)
        loss_gan_real = sigmoid_crossEntropy_withLogit_stable(out_gan_real, gt_gan_real)
        optimizer_gan.zero_grad()
        loss_gan_real.backward()
        optimizer_gan.step()

        if i == batch_per_epoch - 1 or i == 0:
            loss_name_lst = ["loss_class_src_sum",
                               "loss_dis_src",
                               "loss_ht_g_xt_entropy",
                               "loss_gan_fake",
                               "loss_mimic_hs_combine_g_xs",
                               # "loss_mimic_hs_combine_g_xt",
                               # "loss_mimic_ht_g_xt"
                                    ]
            print("epoch {}:\n\tloss_phase_2:\t{:.4f}".format(epoch, loss_phase_2.item()))
            for _n, _l in zip(loss_name_lst, loss_lst):
                print("{}: \t{:.4f}".format(_n, _l.item()))

            for idx, acc_s in enumerate(acc_src_hi_lst):
                print(
                    "\t{}: loss: {:.4f}\tacc: {}".format(source_domains[idx], loss_class_src_lst[idx].item(), acc_s * 100.0))

            print("MIMIC")
            acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
            acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
            acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
            print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
            print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
            print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

            ###### print gan_loss_real
            print("loss_gan_real: {:.4f}".format(loss_gan_real))

def train_phase_3(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                  optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                  num_classes, source_domains, batchnorm_mmd, batch_per_epoch):
    # source_domain_num = len(train_dloader_list[1:])
    generator_model.train()
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
    for (image_t, label_t), (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_trg, dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        optimizer_g.zero_grad()
        optimizer_dis.zero_grad()
        for op_h in classifier_optimizer_list:
            op_h.zero_grad()

        i += 1
        if i == batch_per_epoch:
            break
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]

        latent_trg = generator_model(image_t)
        latent_s_lst = []
        loss_class_src_lst = []

        gt_dis_src = []
        acc_src_hi_lst = []
        for idx in range(len(image_s_lst)):
            latent_s_idx = generator_model(image_s_lst[idx])
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

        output_dis_trg = dis_model(latent_trg)
        loss_dis_trg_entropy = cent(output_dis_trg)

        latent_trg_src = torch.cat([latent_src, latent_trg], dim=0)

        gt_gan_src_real = torch.zeros(latent_src.size(0)).long().cuda()
        gt_gan_src_fake = torch.ones(latent_src.size(0)).long().cuda()

        gt_gan_trg_real = torch.ones(latent_trg.size(0)).long().cuda()
        gt_gan_trg_fake = torch.zeros(latent_trg.size(0)).long().cuda()

        out_gan_fake = gan_model(latent_trg_src)

        gt_gan_fake = torch.cat((gt_gan_src_fake, gt_gan_trg_fake), dim=0)
        loss_gan_fake = sigmoid_crossEntropy_withLogit_stable(out_gan_fake, gt_gan_fake)

        ### hs_combine_xs * c_xs
        output_hs_combine_g_xs = hs_combine_g_x(classifier_list, latent_src, output_dis_src, num_classes)
        gt_hs_combine_g_xs = torch.cat(label_s_lst)
        # print("gt_hs_combine_g_xs.size()", gt_hs_combine_g_xs.size())
        loss_mimic_hs_combine_g_xs = crossEntropy_fn(output_hs_combine_g_xs, gt_hs_combine_g_xs, label_to_1hot=True)
        # print(loss_mimic_hs_combine_g_xs.item())
        ### hs_combine_xt * c_xt
        # with torch.no_grad():
        output_hs_combine_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)

        # loss_mimic_hs_combine_g_xt = crossEntropy_fn(output_hs_combine_g_xt, output_hs_combine_g_xt_argmax,
        #                                              label_to_1hot=True)

        output_ht_g_xt = classifier_list[0](latent_trg)
        output_ht_g_xs = classifier_list[0](latent_src)
        # loss_ht_g_xt_entropy = cent(output_ht_g_xt)
        # output_hs_combine_g_xt_1hot = label_to_1hot(output_hs_combine_g_xt_argmax, num_classes)
        # loss_mimic_ht_g_xt = crossEntropy_fn(softmax_fn(output_ht_g_xt), output_hs_combine_g_xt)
        loss_mimic_ht_g_xt = softmaxCrossEntropy_withLogit(output_ht_g_xt, output_hs_combine_g_xt)
        loss_mimic_ht_g_xs = softmaxcrossEntropy(output_ht_g_xs, gt_hs_combine_g_xs)

        # for param in generator_model.parameters():
        #     param.requires_grad = True

        loss_lst = [loss_class_src_sum,
                           loss_dis_src,
                           # loss_ht_g_xt_entropy,
                           loss_gan_fake,
                           loss_mimic_hs_combine_g_xs,
                           # loss_mimic_hs_combine_g_xt,
                           # loss_mimic_ht_g_xt,
                           loss_mimic_ht_g_xs,
                           ]

        loss_phase_2 = sum(loss_lst)
        loss_phase_2.backward()
        optimizer_g.step()
        optimizer_dis.step()
        for op_h in classifier_optimizer_list:
            op_h.step()

        ### train Gan only #####
        latent_trg = generator_model(image_t)
        latent_s_lst = []
        for idx in range(len(image_s_lst)):
            latent_s_idx = generator_model(image_s_lst[idx])
            latent_s_lst.append(latent_s_idx)

        latent_src = torch.cat(latent_s_lst, dim=0)
        latent_trg_src = torch.cat([latent_src, latent_trg], dim=0)
        out_gan_real = gan_model(latent_trg_src)
        gt_gan_real = torch.cat([gt_gan_src_real, gt_gan_trg_real], dim=0)
        loss_gan_real = sigmoid_crossEntropy_withLogit_stable(out_gan_real, gt_gan_real)
        optimizer_gan.zero_grad()
        loss_gan_real.backward()
        optimizer_gan.step()

        if i == batch_per_epoch - 1 or i == 0:
            loss_name_lst = ["loss_class_src_sum",
                               "loss_dis_src",
                               # "loss_ht_g_xt_entropy",
                               "loss_gan_fake",
                               "loss_mimic_hs_combine_g_xs",
                               # "loss_mimic_hs_combine_g_xt",
                               "loss_mimic_ht_g_xt",
                               "loss_mimic_ht_g_xs"
                                    ]
            print("epoch {}:\n\tloss_phase_2:\t{:.4f}".format(epoch, loss_phase_2.item()))
            for _n, _l in zip(loss_name_lst, loss_lst):
                print("{}: \t{:.4f}".format(_n, _l.item()))

            for idx, acc_s in enumerate(acc_src_hi_lst):
                print(
                    "\t{}: loss: {:.4f}\tacc: {}".format(source_domains[idx], loss_class_src_lst[idx].item(), acc_s * 100.0))

            print("MIMIC")
            acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
            acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
            acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
            print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
            print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
            print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

            ###### print gan_loss_real
            print("loss_gan_real: {:.4f}".format(loss_gan_real))

def train_phase_3_separate(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                  optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                  num_classes, source_domains, batchnorm_mmd, batch_per_epoch):
    # source_domain_num = len(train_dloader_list[1:])
    print("train_phase_3_separate")
    generator_model.train()
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
    for (image_t, label_t), (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_trg, dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        i += 1
        if i == batch_per_epoch:
            break
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]

        # TRAIN PHASE 1
        optimizer_g.zero_grad()
        optimizer_dis.zero_grad()
        for op_h in classifier_optimizer_list:
            op_h.zero_grad()

        latent_trg = generator_model(image_t)
        latent_s_lst = []
        loss_class_src_lst = []

        gt_dis_src = []
        acc_src_hi_lst = []
        for idx in range(len(image_s_lst)):
            latent_s_idx = generator_model(image_s_lst[idx])
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

        # output_dis_trg = dis_model(latent_trg)
        # loss_dis_trg_entropy = cent(output_dis_trg)

        # latent_trg_src = torch.cat([latent_src, latent_trg], dim=0)
        # gt_gan_src_fake = torch.ones(latent_src.size(0)).long().cuda()
        gt_gan_trg_fake = torch.zeros(latent_trg.size(0)).long().cuda()

        # out_gan_fake = gan_model(latent_trg_src)
        #
        # gt_gan_fake = torch.cat((gt_gan_src_fake, gt_gan_trg_fake), dim=0)
        # loss_gan_fake = sigmoid_crossEntropy_withLogit_stable(out_gan_fake, gt_gan_fake)

        out_gan_fake_trg = gan_model(latent_trg)
        loss_gan_fake_trg = sigmoid_crossEntropy_withLogit_stable(out_gan_fake_trg, gt_gan_trg_fake)

        ### hs_combine_xs * c_xs
        output_hs_combine_g_xs = hs_combine_g_x(classifier_list, latent_src, output_dis_src, num_classes)
        gt_hs_combine_g_xs = torch.cat(label_s_lst)
        # print("gt_hs_combine_g_xs.size()", gt_hs_combine_g_xs.size())
        loss_mimic_hs_combine_g_xs = crossEntropy_fn(output_hs_combine_g_xs, gt_hs_combine_g_xs, label_to_1hot=True)
        # print(loss_mimic_hs_combine_g_xs.item())
        ### hs_combine_xt * c_xt
        # with torch.no_grad():
        # output_hs_combine_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)

        # loss_mimic_hs_combine_g_xt = crossEntropy_fn(output_hs_combine_g_xt, output_hs_combine_g_xt_argmax,
        #                                              label_to_1hot=True)

        # output_ht_g_xt = classifier_list[0](latent_trg)
        output_ht_g_xs = classifier_list[0](latent_src)
        # loss_ht_g_xt_entropy = cent(output_ht_g_xt)
        # output_hs_combine_g_xt_1hot = label_to_1hot(output_hs_combine_g_xt_argmax, num_classes)
        # loss_mimic_ht_g_xt = crossEntropy_fn(softmax_fn(output_ht_g_xt), output_hs_combine_g_xt)
        loss_mimic_ht_g_xs = softmaxcrossEntropy(output_ht_g_xs, gt_hs_combine_g_xs)

        # for param in generator_model.parameters():
        #     param.requires_grad = True

        loss_phase1_lst = [loss_class_src_sum,
                           loss_dis_src,
                           # loss_ht_g_xt_entropy,
                           loss_gan_fake_trg,
                           loss_mimic_hs_combine_g_xs,
                           # loss_mimic_hs_combine_g_xt,
                           # 0.1 * loss_mimic_ht_g_xt,
                           loss_mimic_ht_g_xs,
                           ]

        loss_phase_1 = sum(loss_phase1_lst)
        loss_phase_1.backward()

        optimizer_g.step()
        optimizer_dis.step()
        for op_h in classifier_optimizer_list:
            op_h.step()

        ## TRAIN PHASE 2: G + Ht
        generator_model.zero_grad()
        classifier_optimizer_list[0].zero_grad()

        with torch.no_grad():
            latent_trg = generator_model(image_t)
            output_dis_trg = dis_model(latent_trg)
            hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)

        lam = np.random.beta(2, 2)
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
        latent_trg_mixed = generator_model(mixed_image)
        ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
        loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)
        loss_mimic_ht_g_xt_mixed.backward()
        optimizer_g.step()
        classifier_optimizer_list[0].step()

        ### PHASE 3: train Gan only #####

        optimizer_gan.zero_grad()
        with torch.no_grad():
            latent_trg = generator_model(image_t)
            latent_s_lst = []
            for idx in range(len(image_s_lst)):
                latent_s_idx = generator_model(image_s_lst[idx])
                latent_s_lst.append(latent_s_idx)

            latent_src = torch.cat(latent_s_lst, dim=0)
            latent_trg_src = torch.cat([latent_src, latent_trg], dim=0)
        gt_gan_src_real = torch.zeros(latent_src.size(0)).long().cuda()
        gt_gan_trg_real = torch.ones(latent_trg.size(0)).long().cuda()

        out_gan_real = gan_model(latent_trg_src)
        gt_gan_real = torch.cat([gt_gan_src_real, gt_gan_trg_real], dim=0)
        loss_gan_real = sigmoid_crossEntropy_withLogit_stable(out_gan_real, gt_gan_real)
        loss_gan_real.backward()
        optimizer_gan.step()

        if i == batch_per_epoch - 1 or i == 0:
            loss_name_lst = ["loss_class_src_sum",
                               "loss_dis_src",
                               # "loss_ht_g_xt_entropy",
                               "loss_gan_fake_trg",
                               "loss_mimic_hs_combine_g_xs",
                               # "loss_mimic_hs_combine_g_xt",
                               # "loss_mimic_ht_g_xt",
                               "loss_mimic_ht_g_xs"
                                    ]
            print("epoch {}:\n\tloss_phase_1:\t{:.4f}".format(epoch, loss_phase_1.item()))
            for _n, _l in zip(loss_name_lst, loss_phase1_lst):
                print("{}: \t{:.4f}".format(_n, _l.item()))

            for idx, acc_s in enumerate(acc_src_hi_lst):
                print(
                    "\t{}: loss: {:.4f}\tacc: {}".format(source_domains[idx], loss_class_src_lst[idx].item(), acc_s * 100.0))

            print("MIMIC")
            print("loss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed.item()))
            # acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
            # acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
            # acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
            # print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
            # print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
            # print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

            ###### print gan_loss_real
            print("loss_gan_real: {:.4f}".format(loss_gan_real))

def train_phase_3_separate_0(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                  optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                  num_classes, source_domains, batchnorm_mmd, batch_per_epoch):
    # source_domain_num = len(train_dloader_list[1:])
    print("train_phase_3_separate_0, mixup Ht_g_xt with onehot")
    generator_model.train()
    dis_model.train()
    gan_model.train()
    for classifier in classifier_list:
        classifier.train()

    # target_over_src = 5
    dloader_trg = train_dloader[0]
    dloader_src_1 = train_dloader[1]
    dloader_src_2 = train_dloader[2]
    dloader_src_3 = train_dloader[3]
    dloader_src_4 = train_dloader[4]
    dloader_src_5 = train_dloader[5]
    num_src_domain = 5

    i = -1
    for (image_t, label_t), (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_trg, dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        i += 1
        if i == batch_per_epoch:
            break
        image_t = image_t.cuda()
        # label_t = label_t.long().cuda()

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]

        # TRAIN PHASE 1
        optimizer_g.zero_grad()
        optimizer_dis.zero_grad()
        for op_h in classifier_optimizer_list:
            op_h.zero_grad()

        latent_trg = generator_model(image_t)
        latent_s_lst = []
        loss_class_src_lst = []

        gt_dis_src = []
        acc_src_hi_lst = []
        for idx in range(len(image_s_lst)):
            latent_s_idx = generator_model(image_s_lst[idx])
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

        # output_dis_trg = dis_model(latent_trg)
        # loss_dis_trg_entropy = cent(output_dis_trg)

        # latent_trg_src = torch.cat([latent_src, latent_trg], dim=0)
        # gt_gan_src_fake = torch.ones(latent_src.size(0)).long().cuda()
        gt_gan_trg_fake = torch.zeros(latent_trg.size(0)).long().cuda()

        # out_gan_fake = gan_model(latent_trg_src)
        #
        # gt_gan_fake = torch.cat((gt_gan_src_fake, gt_gan_trg_fake), dim=0)
        # loss_gan_fake = sigmoid_crossEntropy_withLogit_stable(out_gan_fake, gt_gan_fake)

        out_gan_fake_trg = gan_model(latent_trg)
        loss_gan_fake_trg = sigmoid_crossEntropy_withLogit_stable(out_gan_fake_trg, gt_gan_trg_fake)

        ### hs_combine_xs * c_xs
        output_hs_combine_g_xs = hs_combine_g_x(classifier_list, latent_src, output_dis_src, num_classes)
        gt_hs_combine_g_xs = torch.cat(label_s_lst)
        # print("gt_hs_combine_g_xs.size()", gt_hs_combine_g_xs.size())
        loss_mimic_hs_combine_g_xs = crossEntropy_fn(output_hs_combine_g_xs, gt_hs_combine_g_xs, label_to_1hot=True)
        # print(loss_mimic_hs_combine_g_xs.item())
        ### hs_combine_xt * c_xt
        # with torch.no_grad():
        # output_hs_combine_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)

        # loss_mimic_hs_combine_g_xt = crossEntropy_fn(output_hs_combine_g_xt, output_hs_combine_g_xt_argmax,
        #                                              label_to_1hot=True)

        # output_ht_g_xt = classifier_list[0](latent_trg)
        output_ht_g_xs = classifier_list[0](latent_src)
        # loss_ht_g_xt_entropy = cent(output_ht_g_xt)
        # output_hs_combine_g_xt_1hot = label_to_1hot(output_hs_combine_g_xt_argmax, num_classes)
        # loss_mimic_ht_g_xt = crossEntropy_fn(softmax_fn(output_ht_g_xt), output_hs_combine_g_xt)
        loss_mimic_ht_g_xs = softmaxcrossEntropy(output_ht_g_xs, gt_hs_combine_g_xs)

        # for param in generator_model.parameters():
        #     param.requires_grad = True

        loss_phase1_lst = [loss_class_src_sum,
                           loss_dis_src,
                           # loss_ht_g_xt_entropy,
                           loss_gan_fake_trg,
                           loss_mimic_hs_combine_g_xs,
                           # loss_mimic_hs_combine_g_xt,
                           # 0.1 * loss_mimic_ht_g_xt,
                           loss_mimic_ht_g_xs,
                           ]

        loss_phase_1 = sum(loss_phase1_lst)
        loss_phase_1.backward()

        optimizer_g.step()
        optimizer_dis.step()
        for op_h in classifier_optimizer_list:
            op_h.step()
        if epoch > 0:
            ## TRAIN PHASE 2: G + Ht
            generator_model.zero_grad()
            classifier_optimizer_list[0].zero_grad()

            with torch.no_grad():
                latent_trg = generator_model(image_t)
                output_dis_trg = dis_model(latent_trg)
                hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)
                # hs_g_xt to one hot: hs_g_xt_onehot
                hs_g_xt_max_by_class, hs_g_xt_arg_max = hs_g_xt.max(1)
                hs_g_xt_onehot = label_to_1hot(hs_g_xt_arg_max, num_classes)
            lam = np.random.beta(2, 2)
            batch_size = image_t.size(0)
            index = torch.randperm(batch_size).cuda()
            mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
            # mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
            mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]
            latent_trg_mixed = generator_model(mixed_image)
            ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
            # loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)
            confidence_gate = (0.95 - 0.8) * (epoch / 60) + 0.8
            if confidence_gate > 0.95:
                confidence_gate = 0.95
            hs_g_xt_mask = (hs_g_xt_max_by_class > confidence_gate).float().cuda()
            # loss_mimic_ht_g_xt_mixed = torch.mean(hs_g_xt_mask * torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))
            loss_mimic_ht_g_xt_mixed = torch.mean(torch.sum(-1.0 * mixed_hs_g_xt_onehot * torch.log_softmax(ht_g_xt_mixed, dim=1), dim=-1))

            loss_mimic_ht_g_xt_mixed.backward()
            optimizer_g.step()
            classifier_optimizer_list[0].step()

        ### PHASE 3: train Gan only #####

        optimizer_gan.zero_grad()
        with torch.no_grad():
            latent_trg = generator_model(image_t)
            latent_s_lst = []
            for idx in range(len(image_s_lst)):
                latent_s_idx = generator_model(image_s_lst[idx])
                latent_s_lst.append(latent_s_idx)

            latent_src = torch.cat(latent_s_lst, dim=0)
            latent_trg_src = torch.cat([latent_src, latent_trg], dim=0)
        gt_gan_src_real = torch.zeros(latent_src.size(0)).long().cuda()
        gt_gan_trg_real = torch.ones(latent_trg.size(0)).long().cuda()

        out_gan_real = gan_model(latent_trg_src)
        gt_gan_real = torch.cat([gt_gan_src_real, gt_gan_trg_real], dim=0)
        loss_gan_real = sigmoid_crossEntropy_withLogit_stable(out_gan_real, gt_gan_real)
        loss_gan_real.backward()
        optimizer_gan.step()

        if i == batch_per_epoch - 1 or i == 0:
            loss_name_lst = ["loss_class_src_sum",
                               "loss_dis_src",
                               # "loss_ht_g_xt_entropy",
                               "loss_gan_fake_trg",
                               "loss_mimic_hs_combine_g_xs",
                               # "loss_mimic_hs_combine_g_xt",
                               # "loss_mimic_ht_g_xt",
                               "loss_mimic_ht_g_xs"
                                    ]
            print("epoch {}:\n\tloss_phase_1:\t{:.4f}".format(epoch, loss_phase_1.item()))
            for _n, _l in zip(loss_name_lst, loss_phase1_lst):
                print("{}: \t{:.4f}".format(_n, _l.item()))

            for idx, acc_s in enumerate(acc_src_hi_lst):
                print(
                    "\t{}: loss: {:.4f}\tacc: {}".format(source_domains[idx], loss_class_src_lst[idx].item(), acc_s * 100.0))
            if epoch > 0:
                print("loss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed.item()))
            # acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
            # acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
            # acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
            # print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
            # print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
            # print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

            ###### print gan_loss_real
            print("loss_gan_real: {:.4f}".format(loss_gan_real))

def train_phase_3_separate_1(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                  optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                  num_classes, source_domains, batchnorm_mmd, batch_per_epoch):
    # source_domain_num = len(train_dloader_list[1:])
    print("train_phase_3_separate_1")
    generator_model.train()
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
    for (image_t, label_t), (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_trg, dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        i += 1
        if i == batch_per_epoch:
            break
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]

        # TRAIN PHASE 1
        optimizer_g.zero_grad()
        optimizer_dis.zero_grad()
        for op_h in classifier_optimizer_list:
            op_h.zero_grad()

        latent_trg = generator_model(image_t)
        latent_s_lst = []
        loss_class_src_lst = []

        gt_dis_src = []
        acc_src_hi_lst = []
        for idx in range(len(image_s_lst)):
            latent_s_idx = generator_model(image_s_lst[idx])
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

        # output_dis_trg = dis_model(latent_trg)
        # loss_dis_trg_entropy = cent(output_dis_trg)

        # latent_trg_src = torch.cat([latent_src, latent_trg], dim=0)
        # gt_gan_src_fake = torch.ones(latent_src.size(0)).long().cuda()
        gt_gan_trg_fake = torch.zeros(latent_trg.size(0)).long().cuda()

        # out_gan_fake = gan_model(latent_trg_src)
        #
        # gt_gan_fake = torch.cat((gt_gan_src_fake, gt_gan_trg_fake), dim=0)
        # loss_gan_fake = sigmoid_crossEntropy_withLogit_stable(out_gan_fake, gt_gan_fake)

        out_gan_fake_trg = gan_model(latent_trg)
        loss_gan_fake_trg = sigmoid_crossEntropy_withLogit_stable(out_gan_fake_trg, gt_gan_trg_fake)

        ### hs_combine_xs * c_xs
        output_hs_combine_g_xs = hs_combine_g_x(classifier_list, latent_src, output_dis_src, num_classes)
        gt_hs_combine_g_xs = torch.cat(label_s_lst)
        # print("gt_hs_combine_g_xs.size()", gt_hs_combine_g_xs.size())
        loss_mimic_hs_combine_g_xs = crossEntropy_fn(output_hs_combine_g_xs, gt_hs_combine_g_xs, label_to_1hot=True)
        # print(loss_mimic_hs_combine_g_xs.item())
        ### hs_combine_xt * c_xt
        # with torch.no_grad():
        # output_hs_combine_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)

        # loss_mimic_hs_combine_g_xt = crossEntropy_fn(output_hs_combine_g_xt, output_hs_combine_g_xt_argmax,
        #                                              label_to_1hot=True)

        # output_ht_g_xt = classifier_list[0](latent_trg)
        output_ht_g_xs = classifier_list[0](latent_src)
        # loss_ht_g_xt_entropy = cent(output_ht_g_xt)
        # output_hs_combine_g_xt_1hot = label_to_1hot(output_hs_combine_g_xt_argmax, num_classes)
        # loss_mimic_ht_g_xt = crossEntropy_fn(softmax_fn(output_ht_g_xt), output_hs_combine_g_xt)
        # loss_mimic_ht_g_xs = softmaxcrossEntropy(output_ht_g_xs, gt_hs_combine_g_xs)

        # for param in generator_model.parameters():
        #     param.requires_grad = True

        loss_phase1_lst = [loss_class_src_sum,
                           loss_dis_src,
                           # loss_ht_g_xt_entropy,
                           loss_gan_fake_trg,
                           loss_mimic_hs_combine_g_xs,
                           # loss_mimic_hs_combine_g_xt,
                           # 0.1 * loss_mimic_ht_g_xt,
                           # loss_mimic_ht_g_xs,
                           ]

        loss_phase_1 = sum(loss_phase1_lst)
        loss_phase_1.backward()

        optimizer_g.step()
        optimizer_dis.step()
        for op_h in classifier_optimizer_list:
            op_h.step()

        ## TRAIN PHASE 2: G + Ht
        generator_model.zero_grad()
        classifier_optimizer_list[0].zero_grad()

        with torch.no_grad():
            latent_trg = generator_model(image_t)
            output_dis_trg = dis_model(latent_trg)
            hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)
            # hs_g_xt to one hot: hs_g_xt_onehot
        lam = np.random.beta(2, 2)
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
        # mixed_hs_g_xt_onehot = lam * hs_g_xt_onehot + (1 - lam) * hs_g_xt_onehot[index, :]
        latent_trg_mixed = generator_model(mixed_image)
        ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
        loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)
        # loss_mimic_ht_g_xt_mixed = torch.mean((hs_g_xt_prob > 0.9) * sum(mixed_hs_g_xt_onehot * log_softmax(ht_g_xt_mixed)))
        loss_mimic_ht_g_xt_mixed.backward()
        optimizer_g.step()
        classifier_optimizer_list[0].step()

        ### PHASE 3: train Gan only #####

        optimizer_gan.zero_grad()
        with torch.no_grad():
            latent_trg = generator_model(image_t)
            latent_s_lst = []
            for idx in range(len(image_s_lst)):
                latent_s_idx = generator_model(image_s_lst[idx])
                latent_s_lst.append(latent_s_idx)

            latent_src = torch.cat(latent_s_lst, dim=0)
            latent_trg_src = torch.cat([latent_src, latent_trg], dim=0)
        gt_gan_src_real = torch.zeros(latent_src.size(0)).long().cuda()
        gt_gan_trg_real = torch.ones(latent_trg.size(0)).long().cuda()

        out_gan_real = gan_model(latent_trg_src)
        gt_gan_real = torch.cat([gt_gan_src_real, gt_gan_trg_real], dim=0)
        loss_gan_real = sigmoid_crossEntropy_withLogit_stable(out_gan_real, gt_gan_real)
        loss_gan_real.backward()
        optimizer_gan.step()

        if i == batch_per_epoch - 1 or i == 0:
            loss_name_lst = ["loss_class_src_sum",
                               "loss_dis_src",
                               # "loss_ht_g_xt_entropy",
                               "loss_gan_fake_trg",
                               "loss_mimic_hs_combine_g_xs",
                               # "loss_mimic_hs_combine_g_xt",
                               # "loss_mimic_ht_g_xt",
                               # "loss_mimic_ht_g_xs"
                                    ]
            print("epoch {}:\n\tloss_phase_1:\t{:.4f}".format(epoch, loss_phase_1.item()))
            for _n, _l in zip(loss_name_lst, loss_phase1_lst):
                print("{}: \t{:.4f}".format(_n, _l.item()))

            for idx, acc_s in enumerate(acc_src_hi_lst):
                print(
                    "\t{}: loss: {:.4f}\tacc: {}".format(source_domains[idx], loss_class_src_lst[idx].item(), acc_s * 100.0))

            print("loss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed.item()))
            # acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
            # acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
            # acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
            # print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
            # print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
            # print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

            ###### print gan_loss_real
            print("loss_gan_real: {:.4f}".format(loss_gan_real))


def train_phase_3_separate_2(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                  optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                  num_classes, source_domains, batchnorm_mmd, batch_per_epoch):
    # source_domain_num = len(train_dloader_list[1:])
    print("train_phase_3_separate_2, mixup to C and Ht")
    generator_model.train()
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
    for (image_t, label_t), (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_trg, dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        i += 1
        if i == batch_per_epoch:
            break
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]

        # TRAIN PHASE 0
        optimizer_g.zero_grad()
        optimizer_dis.zero_grad()
        for op_h in classifier_optimizer_list:
            op_h.zero_grad()

        latent_trg = generator_model(image_t)
        latent_s_lst = []
        loss_class_src_lst = []

        gt_dis_src = []
        acc_src_hi_lst = []
        for idx in range(len(image_s_lst)):
            latent_s_idx = generator_model(image_s_lst[idx])
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
        gt_gan_trg_fake = torch.zeros(latent_trg.size(0)).long().cuda()

        out_gan_fake_trg = gan_model(latent_trg)
        loss_gan_fake_trg = sigmoid_crossEntropy_withLogit_stable(out_gan_fake_trg, gt_gan_trg_fake)

        ### hs_combine_xs * c_xs
        output_hs_combine_g_xs = hs_combine_g_x(classifier_list, latent_src, output_dis_src, num_classes)
        gt_hs_combine_g_xs = torch.cat(label_s_lst)
        # print("gt_hs_combine_g_xs.size()", gt_hs_combine_g_xs.size())
        loss_mimic_hs_combine_g_xs = crossEntropy_fn(output_hs_combine_g_xs, gt_hs_combine_g_xs, label_to_1hot=True)

        # output_ht_g_xt = classifier_list[0](latent_trg)
        output_ht_g_xs = classifier_list[0](latent_src)
        loss_mimic_ht_g_xs = softmaxcrossEntropy(output_ht_g_xs, gt_hs_combine_g_xs)

        loss_phase1_lst = [loss_class_src_sum,
                           loss_dis_src,
                           # loss_ht_g_xt_entropy,
                           loss_gan_fake_trg,
                           loss_mimic_hs_combine_g_xs,
                           # loss_mimic_hs_combine_g_xt,
                           # 0.1 * loss_mimic_ht_g_xt,
                           loss_mimic_ht_g_xs,
                           ]

        loss_phase_1 = sum(loss_phase1_lst)
        loss_phase_1.backward()

        optimizer_g.step()
        optimizer_dis.step()
        for op_h in classifier_optimizer_list:
            op_h.step()

        ## TRAIN PHASE 1: G + C + mixup
        generator_model.zero_grad()
        dis_model.zero_grad()

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
            latent_s_idx = generator_model(mixed_image_s[idx * minibatch: (idx+1)*minibatch])
            latent_s_lst.append(latent_s_idx)
        latent_src = torch.cat(latent_s_lst)
        output_dis_src_mixed = dis_model(latent_src)
        loss_dis_src_mixed = softmaxCrossEntropy_withLogit(output_dis_src_mixed, mixed_gt_dis_model)

        loss_dis_src_mixed.backward()
        optimizer_g.step()
        optimizer_dis.step()

        ## TRAIN PHASE 2: G + Ht
        generator_model.zero_grad()
        classifier_optimizer_list[0].zero_grad()

        with torch.no_grad():
            latent_trg = generator_model(image_t)
            output_dis_trg = dis_model(latent_trg)
            hs_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)

        lam = np.random.beta(2, 2)
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        mixed_hs_g_xt = lam * hs_g_xt + (1 - lam) * hs_g_xt[index, :]
        latent_trg_mixed = generator_model(mixed_image)
        ht_g_xt_mixed = classifier_list[0](latent_trg_mixed)
        loss_mimic_ht_g_xt_mixed = softmaxCrossEntropy_withLogit(ht_g_xt_mixed, mixed_hs_g_xt)
        loss_mimic_ht_g_xt_mixed.backward()
        optimizer_g.step()
        classifier_optimizer_list[0].step()

        ### PHASE 3: train Gan only #####

        optimizer_gan.zero_grad()
        with torch.no_grad():
            latent_trg = generator_model(image_t)
            latent_s_lst = []
            for idx in range(len(image_s_lst)):
                latent_s_idx = generator_model(image_s_lst[idx])
                latent_s_lst.append(latent_s_idx)

            latent_src = torch.cat(latent_s_lst, dim=0)
            latent_trg_src = torch.cat([latent_src, latent_trg], dim=0)
        gt_gan_src_real = torch.zeros(latent_src.size(0)).long().cuda()
        gt_gan_trg_real = torch.ones(latent_trg.size(0)).long().cuda()

        out_gan_real = gan_model(latent_trg_src)
        gt_gan_real = torch.cat([gt_gan_src_real, gt_gan_trg_real], dim=0)
        loss_gan_real = sigmoid_crossEntropy_withLogit_stable(out_gan_real, gt_gan_real)
        loss_gan_real.backward()
        optimizer_gan.step()

        if i == batch_per_epoch - 1 or i == 0:
            loss_name_lst = ["loss_class_src_sum",
                               "loss_dis_src",
                               # "loss_ht_g_xt_entropy",
                               "loss_gan_fake_trg",
                               "loss_mimic_hs_combine_g_xs",
                               # "loss_mimic_hs_combine_g_xt",
                               # "loss_mimic_ht_g_xt",
                               "loss_mimic_ht_g_xs"
                                    ]
            print("epoch {}:\n\tloss_phase_1:\t{:.4f}".format(epoch, loss_phase_1.item()))
            for _n, _l in zip(loss_name_lst, loss_phase1_lst):
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
            print("loss_gan_real: {:.4f}".format(loss_gan_real))


def train_phase_4_separate(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                  optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                  num_classes, source_domains, batchnorm_mmd, batch_per_epoch):
    # source_domain_num = len(train_dloader_list[1:])
    print("train_phase_4_separate, mixup Ht, remove Gan, G-share, C train through hs_combine")

    generator_model.train()
    dis_model.train()
    # gan_model.train()
    for hs_i in classifier_list:
        hs_i.train()

    dloader_trg = train_dloader[0]
    dloader_src_1 = train_dloader[1]
    dloader_src_2 = train_dloader[2]
    dloader_src_3 = train_dloader[3]
    dloader_src_4 = train_dloader[4]
    dloader_src_5 = train_dloader[5]

    i = -1
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

        # TRAIN PHASE 0
        optimizer_g.zero_grad()
        for op_h in classifier_optimizer_list[1:]:
            op_h.zero_grad()

        loss_class_src_lst = []
        for idx in range(len(image_s_lst)):
            latent_s_idx = generator_model(image_s_lst[idx])
            output_hi_g_xi = classifier_list[idx + 1](latent_s_idx)
            task_loss_s = softmaxcrossEntropy(output_hi_g_xi, label_s_lst[idx])
            loss_class_src_lst.append(task_loss_s)
        loss_class_src_sum = sum(loss_class_src_lst)

        loss_class_src_sum.backward()
        optimizer_g.step()
        for op_h in classifier_optimizer_list[1:]:
            op_h.step()
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

        # print dis_model.weight
        # print("Before update C", dis_model.linear_raw[0].weight[0][:10])
        optimizer_dis.zero_grad()
        loss_hs_combine_lst = []

        with torch.no_grad():
            latent_s_lst = []
            for idx in range(len(image_s_lst)):
                latent_s_i = generator_model(image_s_lst[idx])
                latent_s_lst.append(latent_s_i)
            latent_s = torch.cat(latent_s_lst)
            out_hs = []
            for classifier in classifier_list[1:]:
                out_hs_i = classifier(latent_s)  # h
                out_hs.append(softmax_fn(out_hs_i))
        label_s = torch.cat(label_s_lst)
        out_dis_s = dis_model(latent_s)
        output_hs_combine_g_xs = hs_gx_C_gx(out_hs, out_dis_s, num_classes)

        loss_hs_combine = crossEntropy_fn(output_hs_combine_g_xs, label_s, label_to_1hot=True)
        loss_hs_combine.backward()
        optimizer_dis.step()

        # print("After update C", dis_model.linear_raw[0].weight[0][:10])
        # print torch.mean(dis_model.state)dis_model.weight

    mean_confident = AverageMeter()
    for i, (image_t, label_t) in enumerate(dloader_trg):
        i+=1
        if i > batch_per_epoch:
            break
        image_t = image_t.cuda()
        # label_t = label_t.long().cuda()

        ## TRAIN PHASE 2: Gt + Ht mixup
        optimizer_g.zero_grad()
        classifier_optimizer_list[0].zero_grad()

        with torch.no_grad():
            latent_trg = generator_model(image_t)
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
        latent_trg_mixed = generator_model(mixed_image)
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
        optimizer_g.step()
        classifier_optimizer_list[0].step()

    #### MIX Generator\
    print("epoch {}:".format(epoch))
    for n_s, l_s in zip(source_domains, loss_class_src_lst):
        print("\tloss_class_{}: \t{:.4f}".format(n_s, l_s.item()))

    print("\tloss_hs_combine_g_xs: {:.4f}".format(loss_hs_combine.item()))
    print("MIMIC")
    print("\tloss_mimic_ht_g_xt_mixed: {:.4f}".format(loss_mimic_ht_g_xt_mixed.item()))
    print("\tmean_confident_hs_combine_g_xt: {:.4f}".format(mean_confident.avg))


def train_phase_4(train_dloader, generator_model, classifier_list, dis_model, gan_model, optimizer_g,
                  optimizer_dis, optimizer_gan, classifier_optimizer_list, epoch, writer,
                  num_classes, source_domains, batchnorm_mmd, batch_per_epoch):
    # source_domain_num = len(train_dloader_list[1:])
    generator_model.train()
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
    for (image_t, label_t), (image_s_1, label_s_1), (image_s_2, label_s_2), (image_s_3, label_s_3), (
            image_s_4, label_s_4), (image_s_5, label_s_5) \
            in zip(dloader_trg, dloader_src_1, dloader_src_2, dloader_src_3, dloader_src_4, dloader_src_5):
        optimizer_g.zero_grad()
        optimizer_dis.zero_grad()
        for op_h in classifier_optimizer_list:
            op_h.zero_grad()

        i += 1
        if i == batch_per_epoch:
            break
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()

        image_s_lst = [
            image_s_1.cuda(), image_s_2.cuda(), image_s_3.cuda(), image_s_4.cuda(), image_s_5.cuda()
        ]
        label_s_lst = [
            label_s_1.long().cuda(), label_s_2.long().cuda(), label_s_3.long().cuda(), label_s_4.long().cuda(),
            label_s_5.long().cuda()
        ]

        latent_trg = generator_model(image_t)
        latent_s_lst = []
        loss_class_src_lst = []

        gt_dis_src = []
        acc_src_hi_lst = []
        for idx in range(len(image_s_lst)):
            latent_s_idx = generator_model(image_s_lst[idx])
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

        output_dis_trg = dis_model(latent_trg)
        # loss_dis_trg_entropy = cent(output_dis_trg)

        latent_trg_src = torch.cat([latent_src, latent_trg], dim=0)

        gt_gan_src_real = torch.zeros(latent_src.size(0)).long().cuda()
        gt_gan_src_fake = torch.ones(latent_src.size(0)).long().cuda()

        gt_gan_trg_real = torch.ones(latent_trg.size(0)).long().cuda()
        gt_gan_trg_fake = torch.zeros(latent_trg.size(0)).long().cuda()

        out_gan_fake = gan_model(latent_trg_src)

        gt_gan_fake = torch.cat((gt_gan_src_fake, gt_gan_trg_fake), dim=0)
        loss_gan_fake = sigmoid_crossEntropy_withLogit_stable(out_gan_fake, gt_gan_fake)

        ### hs_combine_xs * c_xs
        output_hs_combine_g_xs = hs_combine_g_x(classifier_list, latent_src, output_dis_src, num_classes)
        gt_hs_combine_g_xs = torch.cat(label_s_lst)
        # print("gt_hs_combine_g_xs.size()", gt_hs_combine_g_xs.size())
        loss_mimic_hs_combine_g_xs = crossEntropy_fn(output_hs_combine_g_xs, gt_hs_combine_g_xs, label_to_1hot=True)
        # print(loss_mimic_hs_combine_g_xs.item())
        ### hs_combine_xt * c_xt
        # with torch.no_grad():
        output_hs_combine_g_xt = hs_combine_g_x(classifier_list, latent_trg, output_dis_trg, num_classes)

        loss_mimic_hs_combine_g_xt = crossEntropy_fn(output_hs_combine_g_xt, output_hs_combine_g_xt)

        output_ht_g_xt = classifier_list[0](latent_trg)
        output_ht_g_xs = classifier_list[0](latent_src)
        loss_ht_g_xt_entropy = cent(output_ht_g_xt)
        # output_hs_combine_g_xt_1hot = label_to_1hot(output_hs_combine_g_xt_argmax, num_classes)
        # loss_mimic_ht_g_xt = crossEntropy_fn(softmax_fn(output_ht_g_xt), output_hs_combine_g_xt)
        loss_mimic_ht_g_xt = softmaxCrossEntropy_withLogit(output_ht_g_xt, output_hs_combine_g_xt)
        loss_mimic_ht_g_xs = softmaxcrossEntropy(output_ht_g_xs, gt_hs_combine_g_xs)

        # for param in generator_model.parameters():
        #     param.requires_grad = True

        loss_lst = [loss_class_src_sum,
                           loss_dis_src,
                           loss_ht_g_xt_entropy,
                           loss_gan_fake,
                           loss_mimic_hs_combine_g_xs,
                           loss_mimic_hs_combine_g_xt,
                           loss_mimic_ht_g_xt,
                           loss_mimic_ht_g_xs,
                           ]

        loss_phase_2 = sum(loss_lst)
        loss_phase_2.backward()
        optimizer_g.step()
        optimizer_dis.step()
        for op_h in classifier_optimizer_list:
            op_h.step()

        ### train Gan only #####
        latent_trg = generator_model(image_t)
        latent_s_lst = []
        for idx in range(len(image_s_lst)):
            latent_s_idx = generator_model(image_s_lst[idx])
            latent_s_lst.append(latent_s_idx)

        latent_src = torch.cat(latent_s_lst, dim=0)
        latent_trg_src = torch.cat([latent_src, latent_trg], dim=0)
        out_gan_real = gan_model(latent_trg_src)
        gt_gan_real = torch.cat([gt_gan_src_real, gt_gan_trg_real], dim=0)
        loss_gan_real = sigmoid_crossEntropy_withLogit_stable(out_gan_real, gt_gan_real)
        optimizer_gan.zero_grad()
        loss_gan_real.backward()
        optimizer_gan.step()

        if i == batch_per_epoch - 1 or i == 0:
            loss_name_lst = ["loss_class_src_sum",
                               "loss_dis_src",
                               "loss_ht_g_xt_entropy",
                               "loss_gan_fake",
                               "loss_mimic_hs_combine_g_xs",
                               "loss_mimic_hs_combine_g_xt",
                               "loss_mimic_ht_g_xt",
                               "loss_mimic_ht_g_xs"
                                    ]
            print("epoch {}:\n\tloss_phase_2:\t{:.4f}".format(epoch, loss_phase_2.item()))
            for _n, _l in zip(loss_name_lst, loss_lst):
                print("{}: \t{:.4f}".format(_n, _l.item()))

            for idx, acc_s in enumerate(acc_src_hi_lst):
                print(
                    "\t{}: loss: {:.4f}\tacc: {}".format(source_domains[idx], loss_class_src_lst[idx].item(), acc_s * 100.0))

            print("MIMIC")
            acc_hs_g_xs = compute_acc(output_hs_combine_g_xs, gt_hs_combine_g_xs)
            acc_hs_g_xt = compute_acc(output_hs_combine_g_xt, label_t)
            acc_ht_g_xt = compute_acc(output_ht_g_xt, label_t)
            print("\tacc_hs_g_xs: {}".format(acc_hs_g_xs * 100.0))
            print("\tacc_hs_g_xt: {}".format(acc_hs_g_xt * 100.0))
            print("\tacc_ht_g_xt: {}".format(acc_ht_g_xt * 100.0))

            ###### print gan_loss_real
            print("loss_gan_real: {:.4f}".format(loss_gan_real))

def test_trg(target_domain, source_domains, test_dloader_list, generator_model, dis_model, gan_model, classifier_list,
             epoch, writer,
             num_classes=126, top_5_accuracy=True, test_iter=None, name_summary="Test"):
    source_domain_losses = [AverageMeter() for i in source_domains]
    loss_ht_g_xt = AverageMeter()
    loss_hs_g_xt = AverageMeter()
    acc_ht_g_xt = AverageMeter()
    acc_hs_g_xt = AverageMeter()

    generator_model.eval()
    dis_model.eval()
    gan_model.eval()
    for classifier in classifier_list:
        classifier.eval()
    # calculate loss, accuracy for target domain
    tmp_score = []
    tmp_label = []
    output_dis_model_t_lst = []
    test_dloader_t = test_dloader_list[0]
    for idx, (image_t, label_t) in enumerate(test_dloader_t):
        if test_iter is not None and idx >= test_iter:
            break
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()
        with torch.no_grad():
            latent_t = generator_model(image_t)
            output_dis_model_t = dis_model(latent_t)
            output_ht_g_xt = classifier_list[0](latent_t)
            output_hs_combine_g_xt = hs_combine_g_x(classifier_list, latent_t, output_dis_model_t, num_classes)

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
    output_dis_model_t_lst = torch.mean(softmax_fn(torch.cat(output_dis_model_t_lst)), dim=0)
    print("output_dis_model_t_mean", output_dis_model_t_lst)

    # # calculate loss, accuracy for source domains
    for s_i, domain_s in enumerate(source_domains):
        tmp_score = []
        tmp_label = []
        acc_hs_g_xs = AverageMeter()
        acc_fn = AverageMeter()
        test_dloader_s = test_dloader_list[s_i + 1]
        for it, (image_s, label_s) in enumerate(test_dloader_s):
            if test_iter is not None and it >= test_iter:
                break
            image_s = image_s.cuda()
            label_s = label_s.long().cuda()
            with torch.no_grad():
                latent_s = generator_model(image_s)
                output_s = classifier_list[s_i + 1](latent_s)
                out_dis_model_s = dis_model(latent_s)
                output_hs_combine = hs_combine_g_x(classifier_list, latent_s, out_dis_model_s, num_classes)
            label_onehot_s = torch.zeros(label_s.size(0), num_classes).cuda().scatter_(1, label_s.view(-1, 1), 1)
            task_loss_s = softmaxcrossEntropy(output_s, label_s)
            source_domain_losses[s_i].update(float(task_loss_s.item()), image_s.size(0))
            tmp_score.append(torch.softmax(output_s, dim=1))
            acc_hs_g_xs.update(float(compute_acc(output_hs_combine, label_s)), image_s.size(0))
            acc_fn.update(float(compute_acc(output_s, label_s)), image_s.size(0))
            # turn label into one-hot code
            tmp_label.append(label_onehot_s)
        writer.add_scalar(tag="{}/source_domain_{}_loss".format(name_summary, domain_s),
                          scalar_value=source_domain_losses[s_i].avg,
                          global_step=epoch + 1)
        # tmp_score = torch.cat(tmp_score, dim=0).detach()
        # tmp_label = torch.cat(tmp_label, dim=0).detach()
        # _, y_true = torch.topk(tmp_label, k=1, dim=1)
        # _, y_pred = torch.topk(tmp_score, k=5, dim=1)
        # top_1_accuracy_s = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
        writer.add_scalar(tag="{}/source_domain_{}_acc_top1".format(name_summary, domain_s),
                          scalar_value=acc_fn.avg * 100.0,
                          global_step=epoch + 1)
        print("\t{} \tloss: {:.3f}\tacc: {:.3f}\tacc_hs_g_xs: {:.3f}".format(domain_s,
                                                                             source_domain_losses[s_i].avg,
                                                                             acc_fn.avg * 100.0,
                                                                             acc_hs_g_xs.avg * 100.0,
                                                                             ))


# def test(target_domain, source_domains, test_dloader_list, generator_model, classifier_list, epoch, writer,
#          num_classes=126,
#          top_5_accuracy=True, test_iter=None):
#     source_domain_losses = [AverageMeter() for i in source_domains]
#     target_domain_losses = AverageMeter()
#     task_criterion = nn.CrossEntropyLoss().cuda()
#     # for model in model_list:
#     #     model.eval()
#     # for classifier in classifier_list:
#     #     classifier.eval()
#     # # calculate loss, accuracy for target domain
#     # tmp_score = []
#     # tmp_label = []
#     # test_dloader_t = test_dloader_list[0]
#     # for _, (image_t, label_t) in enumerate(test_dloader_t):
#     #     image_t = image_t.cuda()
#     #     label_t = label_t.long().cuda()
#     #     with torch.no_grad():
#     #         output_t = classifier_list[0](model_list[0](image_t))
#     #     label_onehot_t = torch.zeros(label_t.size(0), num_classes).cuda().scatter_(1, label_t.view(-1, 1), 1)
#     #     task_loss_t = task_criterion(output_t, label_t)
#     #     target_domain_losses.update(float(task_loss_t.item()), image_t.size(0))
#     #     tmp_score.append(torch.softmax(output_t, dim=1))
#     #     # turn label into one-hot code
#     #     tmp_label.append(label_onehot_t)
#     # writer.add_scalar(tag="Test/target_domain_{}_loss".format(target_domain), scalar_value=target_domain_losses.avg,
#     #                   global_step=epoch + 1)
#     # tmp_score = torch.cat(tmp_score, dim=0).detach()
#     # tmp_label = torch.cat(tmp_label, dim=0).detach()
#     # _, y_true = torch.topk(tmp_label, k=1, dim=1)
#     # _, y_pred = torch.topk(tmp_score, k=5, dim=1)
#     # top_1_accuracy_t = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
#     # writer.add_scalar(tag="Test/target_domain_{}_accuracy_top1".format(target_domain).format(target_domain),
#     #                   scalar_value=top_1_accuracy_t,
#     #                   global_step=epoch + 1)
#     # if top_5_accuracy:
#     #     top_5_accuracy_t = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
#     #     writer.add_scalar(tag="Test/target_domain_{}_accuracy_top5".format(target_domain).format(target_domain),
#     #                       scalar_value=top_5_accuracy_t,
#     #                       global_step=epoch + 1)
#     #     print("Target Domain {} Accuracy Top1 :{:.3f} Top5:{:.3f}".format(target_domain, top_1_accuracy_t,
#     #                                                                       top_5_accuracy_t))
#     # else:
#     #     print("Target Domain {} Accuracy {:.3f}".format(target_domain, top_1_accuracy_t))
#     # # calculate loss, accuracy for source domains
#     for s_i, domain_s in enumerate(source_domains):
#         tmp_score = []
#         tmp_label = []
#         test_dloader_s = test_dloader_list[s_i + 1]
#         for it, (image_s, label_s) in enumerate(test_dloader_s):
#             if test_iter is not None and it >= test_iter:
#                 break
#             image_s = image_s.cuda()
#             label_s = label_s.long().cuda()
#             with torch.no_grad():
#                 output_s = classifier_list[s_i + 1](generator_model(image_s))
#             label_onehot_s = torch.zeros(label_s.size(0), num_classes).cuda().scatter_(1, label_s.view(-1, 1), 1)
#             task_loss_s = task_criterion(output_s, label_s)
#             source_domain_losses[s_i].update(float(task_loss_s.item()), image_s.size(0))
#             tmp_score.append(torch.softmax(output_s, dim=1))
#             # turn label into one-hot code
#             tmp_label.append(label_onehot_s)
#         writer.add_scalar(tag="Test/source_domain_{}_loss".format(domain_s), scalar_value=source_domain_losses[s_i].avg,
#                           global_step=epoch + 1)
#         tmp_score = torch.cat(tmp_score, dim=0).detach()
#         tmp_label = torch.cat(tmp_label, dim=0).detach()
#         _, y_true = torch.topk(tmp_label, k=1, dim=1)
#         _, y_pred = torch.topk(tmp_score, k=5, dim=1)
#         top_1_accuracy_s = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
#         writer.add_scalar(tag="Test/source_domain_{}_accuracy_top1".format(domain_s), scalar_value=top_1_accuracy_s,
#                           global_step=epoch + 1)
#         print("\t{} \tloss: {:.3f}\tacc: {:.3f}".format(domain_s, source_domain_losses[s_i].avg,
#                                                         top_1_accuracy_s * 100.0))
#         # if top_5_accuracy:
#         #     top_5_accuracy_s = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
#         #     writer.add_scalar(tag="Test/source_domain_{}_accuracy_top5".format(domain_s), scalar_value=top_5_accuracy_s,
#         #                       global_step=epoch + 1)
