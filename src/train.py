#!/usr/bin/env python
import builtins
import math
import os
import random
import shutil
import argparse
import datetime
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from models import encoder
from training import moco

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from training.trainer import train, eval_knn, eval_pck
from models.correspondence_model import CorrespondenceModel
from dataset import PhotoSketchDataset, MultiEpochsDataLoader
from utils.metrics import MaskMSE

import warnings

from .config import Config

warnings.filterwarnings("ignore")


def main():
    args = Config()
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print(args.world_size, ngpus_per_node)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.seed is not None:
        cudnn.deterministic = True
        random.seed(args.seed + args.gpu)
        torch.manual_seed(args.seed + args.gpu)
        np.random.seed(args.seed + args.gpu)

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print(args.world_size, args.rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}' '{}'".format(args.arch, "cbn" if args.cbn else "orig"))

    # import the original or the conditional BN version of ResNet
    if args.cbn:
        resnet = encoder
    else:
        raise NotImplementedError("Only ResNet with CBN is supported.")


    # train encoder with the default feature map size
    # train warp estimator with larger feature map size (by replacing stride with dilation)
    if args.task == "encoder":
        rswd = [False, False, False]
    elif args.task == "estimator" or args.task == "both":
        if args.feat_size == 16:
            rswd = [False, False, True]
        elif args.feat_size == 32:
            rswd = [False, True, True]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # no need to dilate layer 4 if not using it
    if 4 not in args.layer:
        rswd[2] = False

    # build model
    model = CorrespondenceModel(
        moco.MoCo,
        resnet.__dict__[args.arch],
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        corr_layer=args.layer,
        pretrained_encoder=args.resume_pretrained_encoder,
        replace_stride_with_dilation=rswd,
        feat_size=args.feat_size,
        stn_size=args.stn_size,
        stn_layer=args.stn_layer
    )

    model.stn = nn.SyncBatchNorm.convert_sync_batchnorm(model.stn)
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = {
        "ce": nn.CrossEntropyLoss().cuda(args.gpu),
        "ce_none": nn.CrossEntropyLoss(reduction="none").cuda(args.gpu),
        "mask_mse": MaskMSE().cuda(args.gpu),  # this MSE does not care about out-of-range values
    }

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.module.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.module.parameters(), args.lr,
                                     weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    if args.rank == 0:
        os.makedirs(os.path.join(args.log_dir, 'runs', args.writer_name), exist_ok=True)
        writer = SummaryWriter(os.path.join(args.log_dir, 'runs/', args.writer_name, args.name))
    else:
        writer = None

    # optionally resume from a checkpoint
    if args.resume or args.resume_encoder:
        resume = args.resume if args.resume else args.resume_encoder
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            if args.gpu is None:
                checkpoint = torch.load(resume)
            else:
                # Load model at specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(resume, map_location=loc)
            state_dict = checkpoint['state_dict']
            args.start_epoch = checkpoint['epoch']

            # if resume(load) just the encoder, pop STN from checkpoint
            if args.resume_encoder:
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if "stn" in k or "pos_map" in k:
                        del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)

            # check if loading successes
            if args.resume:
                assert len(msg.missing_keys) == 0 and len(msg.unexpected_keys) == 0
            elif args.resume_encoder:
                assert len(msg.unexpected_keys) == 0
                for key in msg.missing_keys:
                    assert "stn" in key or "pos_map" in key

            # load optimizer and scaler checkpoint
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scaler.load_state_dict(checkpoint['scaler'])
            except ValueError:
                print("=> fail to load optimizer state_dict")

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(True)
    train_csv = os.path.join(args.csv_path, "train_pairs_ps.csv")
    test_csv = os.path.join(args.csv_path, "test_pairs_ps.csv")

    # build train dataset
    # feature encoder training setting: instance supervision (for ablation) or pair supervision
    if args.supervision == "instance" or args.supervision == "pair":
        train_dataset = PhotoSketchDataset(train_csv, args.data_path, mode="train", mix_within_class=False)
    # class supervision
    elif args.supervision == "class":
        train_dataset = PhotoSketchDataset(train_csv, args.data_path, mode="train", mix_within_class=True)
    else:
        raise NotImplementedError

    # build eval dataset (only for visualization during training)
    eval_dataset = PhotoSketchDataset(test_csv, args.data_path, mode="eval", mix_within_class=False)
    # build test dataset (for error metric computation)
    test_dataset = PhotoSketchDataset(test_csv, args.data_path, mode="test", mix_within_class=False)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True, persistent_workers=True)

    eval_loader = MultiEpochsDataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False, persistent_workers=True)

    test_loader = MultiEpochsDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False, persistent_workers=True)

    args.epochs = args.epochs + 1
    args.epochs = args.epochs + args.start_epoch
    if args.break_training:
        args.break_training = args.break_training + args.start_epoch + 1

    for epoch in range(args.start_epoch, args.break_training if args.break_training else args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)


        lr = adjust_learning_rate(optimizer, epoch, args)
        if writer is not None:
            writer.add_scalar('metric/LR', lr, epoch)

        # train for one epoch
        mem = train(train_loader, model, criterion, optimizer, scaler, epoch, args, writer)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):

            # clone the network for evaluation
            if (epoch % args.knn_freq == 0) or (epoch % args.pck_freq == 0):
                rswd = [False, True, False]

                eval_model = CorrespondenceModel(
                    moco.MoCo,
                    resnet.__dict__[args.arch],
                    dim=args.moco_dim,
                    K=args.moco_k,
                    m=args.moco_m,
                    T=args.moco_t,
                    corr_layer=args.layer,
                    replace_stride_with_dilation=rswd,
                    feat_size=args.feat_size,
                    stn_size=args.stn_size,
                    stn_layer=args.stn_layer
                )
                state_dict = model.state_dict()
                for key, value in list(state_dict.items()):
                    state_dict[key.replace("module.", "")] = value
                    state_dict.pop(key)
                msg = eval_model.load_state_dict(state_dict)
                print("construct evaluation model:", msg)
                eval_model.cuda(args.gpu).eval()

            plot = True if epoch % args.plot_freq == 0 else False
            if epoch % args.knn_freq == 0:
                eval_knn(eval_model, eval_loader, epoch, mem, args, writer, plot=plot)

            if epoch % args.pck_freq == 0:
                eval_pck(eval_model, test_loader, epoch, args, writer)

            if epoch % args.save_freq == 0:
                model_dir = os.path.join(args.save_dir, args.name)
                Path(model_dir).mkdir(parents=True, exist_ok=True)
                name = os.path.join(model_dir, 'checkpoint_{:04d}.pth.tar')
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict()
                }, is_best=False, filename=name.format(epoch))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    lr = args.lr

    if epoch - args.start_epoch + 1 < args.warmup_epochs:
        lr = args.lr * (epoch - args.start_epoch + 1) / args.warmup_epochs
    else:
        if args.cos:
            lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.start_epoch - args.warmup_epochs) /
                                                (args.epochs - args.start_epoch - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    main()
