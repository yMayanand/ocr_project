from glob import glob
import pandas as pd
import torch
from utils import read_image


class OCRDataset(torch.utils.data.Dataset):
    r"""Dataset to train an OCR model
    """

    def __init__(self, root_dir, transforms=None, return_len=False):
        super().__init__()
        self.transforms = transforms
        self.image_files = glob(f"{root_dir}/*")
        self.return_len = return_len
        self.make_vocab()
        self.char2idx = {key: val for val, key in enumerate(self.idx2char)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img = read_image(fname)
        tmp = fname.split('/')[-1].split('.jpg')[0]
        label = list(tmp)

        if self.transforms is not None:
            img = self.transforms(img)
            label = [self.char2idx[char] for char in label]
            length = len(label)
            labels = (label, length) if self.return_len else (label)

        return img, *labels

    def make_vocab(self):
        self.idx2char = set()
        for file in self.image_files:
            tmp = file.split('/')[-1].split('.jpg')[0]
            self.idx2char.update(list(tmp))
        self.idx2char = list(self.idx2char)
        self.idx2char.insert(0, 'BLANK')
        
    def get_characteristics(self):
        df = pd.DataFrame(columns=['height', 'width', 'channels'])
        for file in self.image_files:
            h, w, c = read_image(file).shape
            df.loc[len(df.index)] = [h, w, c]
        return df

