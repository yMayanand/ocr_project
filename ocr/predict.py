from ocr_dataset import OCRDataset
from model import CRNN
from argparse import ArgumentParser
from torchvision import transforms
import torch
from utils import greedy_decode, idx2char, post_process
import random
import numpy as np

parser = ArgumentParser()

parser.add_argument(
    '--ckpt_path',
    type=str,
    help="path of the checkpoint stored"
)

parser.add_argument(
    '--root_dir',
    type=str,
    help="directory where data is stored"
)

parser.add_argument(
    '--idx',
    type=int,
    help="index of dataset on which we want to predict"
)

args = parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Resize((80, 128)),
    transforms.Lambda(lambda x: x/255),
    transforms.Normalize(0.5, 0.5)
])


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

dataset = OCRDataset(args.root_dir, transforms=transform)
image, *target = dataset[args.idx]
target = idx2char(target, dataset)
model = CRNN(80, 1, 36, 256)
state_dict = torch.load(args.ckpt_path)
model.load_state_dict(state_dict['weights'])
model.eval()
preds = model(image.unsqueeze(0))
preds = post_process(preds[:, 0, :], dataset)
print(f"target {''.join(target)}")
print(f"preds {greedy_decode(preds)}")
