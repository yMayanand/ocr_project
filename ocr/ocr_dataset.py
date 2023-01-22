from glob import glob
import pandas as pd
import torch
from utils import read_image

VOCAB_PATCH = [
    'BLANK', 'Z', 'B', '4', 'X', 'R', '2', 'U', 'D', 'G', 
    'Q', 'S', 'A', 'N', 'K', '0', 'C', 'J', 'P', 'Y', 'H', 
    '7', 'W', 'V', '5', 'F', 'L', '8', '1', 'I', 'T', 'M', 
    '3', 'O', '9', 'E', '6']


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
        """self.idx2char = []
        for file in self.image_files:
            tmp = file.split('/')[-1].split('.jpg')[0]
            for i in tmp:
                if i not in self.idx2char:
                    self.idx2char.append(i)
        
        self.idx2char.insert(0, 'BLANK')"""
        self.idx2char = VOCAB_PATCH
        
    def get_characteristics(self):
        df = pd.DataFrame(columns=['height', 'width', 'channels'])
        for file in self.image_files:
            h, w, c = read_image(file).shape
            df.loc[len(df.index)] = [h, w, c]
        return df

