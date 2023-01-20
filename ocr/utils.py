import cv2
import torch
from itertools import groupby
from Levenshtein import distance
from torch.utils.data import random_split

def idx2char(preds, dataset):
    return [dataset.idx2char[idx] for idx in preds]

def post_process(preds, dataset):
    # preds shape (seq_len, num_class)
    _, preds = torch.max(preds, dim=1)
    return idx2char(preds.tolist(), dataset)

def read_image(file):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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

def get_dataset_split(dataset, ratio):
    assert sum(ratio) == 1, "ratio of splits should add up to 1.0"
    train_ds, val_ds = random_split(dataset, ratio)
    return train_ds, val_ds

def custom_collate(batch):
    image, target, target_len = zip(*batch)
    image = torch.stack(image)
    target_len = torch.LongTensor(target_len)
    temp = []
    for i in target:
        temp += i
    target = temp
    target = torch.LongTensor(target)
    return image, target, target_len

def greedy_decode(preds):
    # collapse best path (using itertools.groupby), map to chars, join char list to string
    best_chars_collapsed = [k for k, _ in groupby(preds) if k != 'BLANK']
    res = ''.join(best_chars_collapsed)
    return res


def char_accuracy(preds, ground_truth):
    dist = distance(preds, ground_truth)
    gt_len = len(ground_truth)

    fraction = dist / gt_len
    percent = fraction * 100
    if 100.0 - percent < 0:
            return 0.0
    else:
        return 100.0 - percent

def batch_to_device(batch, device):
    return [item.to(device) for item in batch]
