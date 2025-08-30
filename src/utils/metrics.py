import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskMSE(nn.Module):
    """
    A masked version of MSE, designed for flow.
    It only cares about valid pixels, and ignores errors at pixels with out-of-bound values.
    Ignore conditions: value == 0: typically caused by zero padding in grid_sample
                       value > 1.0 or value < -1.0: typically happened when flow indices went out of range

    """
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        mse = F.mse_loss(input, target, reduction="none")

        mask = torch.ones(mse.shape, dtype=torch.bool).to(mse.device)
        input_mask = input != 0
        target_mask = target != 0
        mask = mask & input_mask & target_mask

        input_mask = torch.abs(input) <= 1
        target_mask = torch.abs(target) <= 1
        mask = mask & input_mask & target_mask
        mask = mask.detach().flatten()

        mse = mse.flatten()
        return mse[mask].mean()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', category="train"):
        self.name = name
        self.category = category
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def log(self, writer, n):
        writer.add_scalar(self.category + "/" + self.name, self.val, n)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def proj_kps(flows, kps, image_size):
    """
    Project src keypoints to the dst image using estimated flows.
    :param flows: Shape: N, H, W, 2
    :param kps: Shape: N, 8, 2
    :param image_size: Size of image. Default: 256.
    :return: The projected keypoints.
    """
    N, H, W, _ = flows.shape
    flows = (flows + 1) * (image_size - 1) / 2
    flows = flows.reshape(N, H * W, 2)
    kps = torch.round(kps).long()
    kps = kps.reshape(N, 8, 2)
    kps = kps[:, :, 0] * image_size + kps[:, :, 1]
    kps = kps.unsqueeze(-1).expand(-1, -1, 2)
    dst = torch.gather(flows, 1, kps).flip(dims=(-1,))
    return dst


def compute_pck(gt_kps, pred_kps, image_size):
    """
    Compute PCK@15, PCK@0.10, PCK@0.05
    :param gt_kps: groundtruth keypoints annotation
    :param pred_kps: predicted keypoints
    :param image_size: size of image
    :return: PCK@10, PCK@5
    """
    diff = np.sqrt(np.sum(np.square(np.array(gt_kps) - np.array(pred_kps)), axis=-1))  # N, 8
    pck10 = np.sum(diff <= image_size * 0.1, axis=1) / 8
    pck05 = np.sum(diff <= image_size * 0.05, axis=1) / 8
    return pck10, pck05
