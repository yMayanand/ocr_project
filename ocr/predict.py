from ocr_dataset import OCRDataset
from model import CRNN
from argparse import ArgumentParser
from torchvision import transforms
import torch
from utils import greedy_decode, idx2char, post_process, read_image
import random
import numpy as np
import matplotlib.pyplot as plt

parser = ArgumentParser()

parser.add_argument(
    '--ckpt_path',
    type=str,
    default=None,
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

parser.add_argument(
    '--img_path',
    type=str,
    help="path to image on which prediction is to be made"
)

args = parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Resize((32, 128)),
    transforms.Lambda(lambda x: x/255),
    transforms.Normalize(0.5, 0.5)
])


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

dataset = OCRDataset(args.root_dir, transforms=transform)
#image, *target = dataset[args.idx]
#target = idx2char(target, dataset)
model = CRNN(80, 1, 37, 256)
if args.ckpt_path is not None:
    state_dict = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['weights'])
model.eval()
image = read_image(args.img_path)
image = transform(image)
plt.imshow(image.permute(1, 2, 0))
plt.savefig('test.jpg')
data = image.unsqueeze(0)
preds = model(data)
preds = post_process(preds[:, 0, :], dataset)
print(preds)
#print(f"target {''.join(target)}")
print(f"preds {greedy_decode(preds)}")
