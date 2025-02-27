import torch
from torchmetrics.functional import accuracy as torch_accuracy


class AverageMeter(object):
    def __init__(self, name, fmt=":f", category="train"):
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
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        batch_size = target.size(0)
        res = []

        for k in topk:
            acc = torch_accuracy(
                output.float(),
                target,
                task="multiclass",
                num_classes=output.size(1),
                top_k=k,
            )
            res.append(acc.mul_(100.0))
        return res
