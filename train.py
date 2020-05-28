import multiprocessing as mp
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn')

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import models
from models import ParameterClient
from logger import create_logger
from datasets import FileListDataset, DistSequentialSampler
from utils import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

classifier_types = sorted(name for name in models.__factory_classifier__)

parser = argparse.ArgumentParser(
    description='PyTorch Face Classification Training')
parser.add_argument('--arch',
                    '-a',
                    metavar='ARCH',
                    default='resnet50',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('--train-filelist', type=str)
parser.add_argument('--train-prefix', type=str)
parser.add_argument('--val-filelist', type=str)
parser.add_argument('--val-prefix', type=str)
parser.add_argument('-j',
                    '--workers',
                    default=0,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs',
                    default=30,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--test-batch-size', default=None, type=int)
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.01,
                    type=float,
                    metavar='LR',
                    help='initial learning rate')
parser.add_argument('--lr-steps',
                    default=[21, 27],
                    type=list,
                    help='stpes to change lr')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
parser.add_argument('--gamma',
                    default=0.1,
                    type=float,
                    help='learing rate multiplier')
parser.add_argument('--input-size',
                    default=112,
                    type=int,
                    help='input size (default: 112x112)')
parser.add_argument('--feature-dim',
                    default=256,
                    type=int,
                    metavar='D',
                    help='feature dimension (default: 256)')
parser.add_argument('--num-classes',
                    default=1000,
                    type=int,
                    metavar='N',
                    help='number of classes (default: 1000)')
parser.add_argument('--sample-num',
                    default=1000,
                    type=int,
                    help='sampling number of classes out of all classes')
parser.add_argument('--print-freq',
                    default=100,
                    type=int,
                    help='logger.info frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-path',
                    default='checkpoints/ckpt',
                    type=str,
                    help='path to store checkpoint (default: checkpoints)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--sampled',
                    dest='sampled',
                    action='store_true',
                    help='sampling from full softmax')
parser.add_argument('--classifier-type',
                    default='linear',
                    choices=classifier_types,
                    help='choose different type of classifier')
parser.add_argument('--distributed',
                    dest='distributed',
                    action='store_true',
                    help='distributed training')
parser.add_argument('--dist-addr',
                    default='127.0.0.1',
                    type=str,
                    help='distributed address')
parser.add_argument('--dist-port',
                    default='23456',
                    type=str,
                    help='distributed port')
parser.add_argument('--dist-backend',
                    default='nccl',
                    type=str,
                    help='distributed backend')
parser.add_argument('--tmp-client-id',
                    default=9999,
                    type=int,
                    help='tmp client used to communicate with paramserver')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # init dist
    gpu_num = torch.cuda.device_count()
    if args.distributed:
        args.rank, args.world_size = init_processes(args.dist_addr,
                                                    args.dist_port, gpu_num,
                                                    args.dist_backend)
        print("=> using {} GPUS for distributed training".format(
            args.world_size))
    else:
        args.rank = 0
        print("=> using {} GPUS for training".format(gpu_num))

    # create logger
    if args.rank == 0:
        mkdir_if_no_exist(args.save_path,
                          subdirs=['events/', 'logs/', 'checkpoints/'])
        tb_logger = SummaryWriter('{}/events'.format(args.save_path))
        logger = create_logger('global_logger',
                               '{}/logs/log.txt'.format(args.save_path))
        logger.debug(args)  # log args only to file
    else:
        tb_logger = None
        logger = None

    # init data loader
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.25, 0.25, 0.25])
    train_dataset = FileListDataset(
        args.train_filelist, args.train_prefix,
        transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = FileListDataset(
        args.val_filelist, args.val_prefix,
        transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        val_sampler = DistSequentialSampler(val_dataset, args.world_size,
                                            args.rank)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler)

    if args.test_batch_size is None:
        args.test_batch_size = 2 * args.batch_size
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.test_batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             sampler=val_sampler)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](feature_dim=args.feature_dim)

    if args.sampled:
        if args.rank > 0:
            assert args.distributed
        assert args.sample_num <= args.num_classes
    model = models.build_classifier(args.classifier_type, model,
                                    **args.__dict__)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, [args.rank])
        print('create DistributedDataParallel model successfully', args.rank)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            args.start_epoch, best_prec1 = load_ckpt(args.resume,
                                                     model,
                                                     optimizer=optimizer)
            if args.sampled:
                with ParameterClient(args.tmp_client_id) as client:
                    cls_resume = args.resume.replace('.pth.tar', '_cls.h5')
                    if os.path.isfile(cls_resume):
                        client.resume(cls_resume)
                        print("=> loaded checkpoint '{}' (epoch {})".format(
                            cls_resume, checkpoint['epoch']))
                    else:
                        print("=> no checkpoint found at '{}'".format(
                            cls_resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, model, criterion, args.print_freq, args.rank,
                 logger, args.sampled)
        return

    assert max(args.lr_steps) < args.epochs
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.lr_steps, args.gamma)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch,
              args.print_freq, args.rank, logger, tb_logger, args.sampled)
        lr_scheduler.step()

        # evaluate on validation set
        prec1, loss = validate(val_loader, model, criterion, args.print_freq,
                               args.rank, logger, args.sampled)

        # remember best prec@1 and save checkpoint
        if args.rank == 0:
            if tb_logger is not None:
                tb_logger.add_scalar('test_acc', prec1, epoch)
                tb_logger.add_scalar('test_loss', loss, epoch)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_ckpt(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }, args.save_path, epoch + 1, is_best)
            if args.sampled:
                with ParameterClient(args.tmp_client_id) as client:
                    client.snapshot('{}_epoch_{}_cls.h5'.format(
                        args.save_path, epoch + 1))


def train(train_loader,
          model,
          criterion,
          optimizer,
          epoch,
          print_freq,
          rank,
          logger,
          tb_logger=None,
          sampled=None):
    batch_time = AverageMeter(10)
    data_time = AverageMeter(10)
    losses = AverageMeter(10)
    top1 = AverageMeter(10)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # target = target.cuda(non_blocking=True)
        target = target.cuda()

        # compute output
        if not sampled:
            output = model(input, target)
        else:
            output, target = model(input, target)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, = accuracy(output, target, topk=(1, ))
        losses.update(loss.item())
        top1.update(prec1[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 and rank == 0 and logger is not None:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'LR: {3}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            epoch,
                            i,
                            len(train_loader),
                            optimizer.param_groups[0]['lr'],
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            top1=top1))
            if tb_logger is not None:
                _iter = epoch * len(train_loader) + i
                tb_logger.add_scalar('train_acc', top1.avg, _iter)
                tb_logger.add_scalar('train_loss', losses.avg, _iter)


def validate(val_loader,
             model,
             criterion,
             print_freq,
             rank,
             logger,
             sampled=None):
    n = len(val_loader)
    batch_time = AverageMeter(10)
    losses = AverageMeter(n)
    top1 = AverageMeter(n)

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            if not sampled:
                output = model(input, target)
            else:
                output, target = model(input, target)
            loss = criterion(output, target)

            prec1, = accuracy(output, target, topk=(1, ))
            losses.update(loss.item())
            top1.update(prec1[0])

            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0 and rank == 0 and logger is not None:
                logger.info(
                    'Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1))
        if rank == 0:
            logger.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg


if __name__ == '__main__':
    main()
