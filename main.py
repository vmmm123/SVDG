import torchvision
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import StepLR
from data_loader import *
from torchvision import transforms
from model import *
from util import *
import argparse


def main(args):
    for time in range(0,args.times):
            target=args.target
            if target =='all':
                targets=args.all[:]
            else:
                targets=[target]

            for target in targets:
                source=args.all[:]
                source.remove(target)
                print("Target domain: {}, Training domans: {}, {}, {}\n".format(target, source[0], source[1], source[2]))
                checkpoint = ModelCheckpoint(mode='max', directory1=args.checkpoint_dir)
                train_loader, valid_loader, target_loader,datasets_len=loader(args.data_dir,source,target,batch_size=args.batch_size,num_domains=args.num_domains)
                net = Network(num_classes=args.num_classes, num_features=args.num_reduced_festures).to(args.device)
                model_lr= [(net.features, 1.0), (net.classifier, 1.0),
                        (net.class_classifier, 1.0 * args.fc_weight), (net.discriminator, 1.0 * args.disc_weight),
                           (net.projection_original_features, 1.0 * args.proj_weight)]
                optimizers = [get_optimizer(model_part, args.lr * i, args.momentum, args.weight_decay) for model_part, i in model_lr]
                schedulers = [StepLR(optimizer=opt, step_size=args.lr_step, gamma=args.lr_decay_gamma)
                                     for opt in optimizers]

                memorys = []
                for i in range(args.num_domains):
                    memorys.append(Memory(size = datasets_len[i], weight= 0.5, device = args.device,num_features=args.num_reduced_festures))
                    memorys[i].initialize(i, net, train_loader)


                state_dict = torch.load("../alexnet_caffe.pth.tar")
                del state_dict["classifier.fc8.weight"]
                del state_dict["classifier.fc8.bias"]
                net.features.load_state_dict(state_dict, strict=False)
                net.classifier.load_state_dict(state_dict, strict=False)

                for epoch in range(0,args.epochs):

                    print('\nEpoch: {}'.format(epoch))
                    net.train()
                    net=train(train_loader, net,optimizers,schedulers,memorys,args)

                    net.eval()
                    val_acc = validate(valid_loader,net,args.device)
                    # save model if improved

                    test_acc=validate(target_loader,net,args.device)

                    print("val_acc:{:.4f}\ttest_acc:{:.4f}\n".format(val_acc,test_acc))
                    checkpoint.save_model(net, val_acc, test_acc, epoch, time, target)

                print("Test accuarcy:{:.4f} on {} by the best model at epoch {}:\n".format(checkpoint.test_best,target,checkpoint.epoch_best))


def train(train_loader, net,optimizers ,schedulers,memorys,args):
    for memory in memorys:
        memory.update_weighted_count()
    train_loss = AverageMeter('train_loss')
    cls_loss = AverageMeter('cls_loss')
    domain_loss = AverageMeter('domain_loss')
    loss_image = AverageMeter('loss_image')
    loss_image2 = AverageMeter('loss_image2')
    loss_entropy = AverageMeter('loss_entropy ')
    noise_contrastive_estimator = NoiseContrastiveEstimator(args.device)
    class_criterion = nn.CrossEntropyLoss().to(args.device)
    entropy_criterion = HLoss().to(args.device)


    for step, batch in enumerate(train_loader):
        x_1, x_2, x_3, idx_1, idx_2, idx_3, o_x_1, o_x_2, o_x_3, y_task_1, y_task_2, y_task_3, y_domain_1, y_domain_2, y_domain_3 = batch
        length = len(idx_1)
        images = torch.cat((x_1, x_2, x_3), dim=0)
        y_task = torch.cat((y_task_1, y_task_2, y_task_3), dim=0)
        y_domain = torch.cat((y_domain_1, y_domain_2, y_domain_3), dim=0)
        index = [idx_1, idx_2, idx_3]
        images2 = torch.cat((o_x_1, o_x_2, o_x_3), dim=0)

        # prepare batch
        images = images.to(args.device)
        images2 = images2.to(args.device)
        y_task = y_task.to(args.device)
        y_domain = y_domain.to(args.device)
        for optimizer in optimizers:
            optimizer.zero_grad()

        # loss backward
        output = net(images=images, images2=images2, mode=1)

        loss_1 = 0
        loss_2 = 0
        for i in range(args.num_domains):
            representations = memorys[i].return_representations(index[i]).to(args.device).detach()
            loss_1 += noise_contrastive_estimator(i, args.temperature, representations, output[1][i * length:(i + 1) * length],
                                                  index[i],
                                                  memorys, num_features=args.num_reduced_festures, negative_p=args.negative_p,
                                                  num_domains=args.num_domains)
            loss_2 += noise_contrastive_estimator(i, args.temperature, representations, output[0][i * length:(i + 1) * length],
                                                  index[i],
                                                  memorys, num_features=args.num_reduced_festures, negative_p=args.negative_p,
                                                  num_domains=args.num_domains)
        loss_cls = class_criterion(output[2], y_task)
        loss_ent = entropy_criterion(output[2])
        loss_domain = class_criterion(output[3], y_domain)
        loss = args.alpha * (loss_1 + loss_2) + args.beta * loss_domain + loss_cls + loss_ent

        loss.backward()

        for optimizer in optimizers:
            optimizer.step()

        # update representation memory
        for i in range(args.num_domains):
            memorys[i].update(index[i], output[0][i * length:(i + 1) * length].detach().cpu().numpy())

        # update metric and bar
        train_loss.update(loss.item(), images.shape[0])
        loss_image.update(loss_1.item(), images.shape[0])
        loss_image2.update(loss_2.item(), images.shape[0])
        loss_entropy.update(loss_ent.item(), images.shape[0])
        cls_loss.update(loss_cls.item(), images.shape[0])
        domain_loss.update(loss_domain.item(), images.shape[0])

    for scheduler in schedulers:
        scheduler.step()
    print("train_loss:{:.4f}\t".format(train_loss.return_avg()))
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../kfold/',help='data folder')
    parser.add_argument('--checkpoint_dir', default='models',help='folder to save model')
    parser.add_argument('--target', default='sketch', help='the target domain of all target domains', choices=['photo', 'cartoon', 'sketch', 'art_painting','all'])
    parser.add_argument('--gpu', type=int, default=0,help='GPU id')
    parser.add_argument('--num-domains', type=int, default=3, help='number of source domians')
    parser.add_argument('--all', default=['photo', 'cartoon', 'sketch', 'art_painting'], help='all domains')
    parser.add_argument('--num_classes', type=int, default=7, help='number of categories')
    parser.add_argument('--negative_p', type=float, default=0.1, help='proportion of negative examples in NCE')
    parser.add_argument('--temperature', type=float, default=0.07, help='temperature in NCE')
    parser.add_argument('--num-reduced-festures', type=int, default=256, help='dimension number of projected features')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for each source domain')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epoches')
    parser.add_argument('--alpha', type=float, default=0.5, help='the weight of self_supeivisor')
    parser.add_argument('--beta', type=float, default=0.5, help='the weight of domain adv')
    parser.add_argument('--lr_step', type=int, default=10, help='number of steps the learning rate decays')
    parser.add_argument('--lr_decay-gamma', type=float, default=0.2, help='ratio of learning rate decay')
    parser.add_argument('--proj_weight', type=float, default=10.0, help='multiple of learning rate in projection network')
    parser.add_argument('--fc_weight', type=float, default=10.0, help='multiple of learning rate in the last fc layer of classifier')
    parser.add_argument('--disc_weight', type=float, default=10.0, help='multiple of learning rate in doamin discriminator')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--times', type=int, default=1, help='times of repetition')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")


    main(args)