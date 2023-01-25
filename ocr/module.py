import math
import torch
import torch.nn.functional as F
from ocr_dataset import OCRDataset
from utils import *
from torch.utils import data
from torchvision import transforms
from torchmetrics import CharErrorRate

class Engine:
    def __init__(self, model, args):
        self.model = model
        self.loss_fn = torch.nn.CTCLoss(zero_infinity=True)
        self.cer_metric_fn = CharErrorRate()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize((32, 128)),
            #transforms.Lambda(lambda x: x/255),
            transforms.Normalize(0.5, 0.5)
        ])

        self.dataset = OCRDataset(
            args.root_dir, 
            transforms=transform, 
            return_len=True
        )

        train_ds, val_ds = get_dataset_split(self.dataset, (0.8, 0.2))
        print('train_ds', len(train_ds))
        print('val_ds', len(val_ds))

        self.train_dl = data.DataLoader(
            train_ds, batch_size=64, 
            shuffle=True, collate_fn=custom_collate,
            num_workers=2
        )

        self.val_dl = data.DataLoader(
            val_ds, batch_size=64, 
            collate_fn=custom_collate,
            num_workers=2
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.wd
        )

    def train_step(self, batch):
        image, target, target_len = batch
        preds = F.log_softmax(self.model(image), 2)
        seq_len, batch_size = preds.shape[:2]
        preds_len = torch.LongTensor([seq_len]*batch_size)
        loss = self.loss_fn(preds, target, preds_len, target_len)
        if math.isnan(loss.item()):
                    print("logits:", preds)
                    print("logits_shape:", preds.shape)
                    print("labels:", target)
                    print("labels_shape", target.shape)
                    print("prediction_sizes:", preds_len)
                    print("target_sizes:", target_len)
                    print("prediction_sizes_shape:", preds_len.shape)
                    print("target_sizes_shape:", target_len.shape)
                    raise Exception("NaN loss obtained. But why?")
        return loss

    def val_step(self, batch):
        with torch.no_grad():
            self.model.eval()
            image, target, target_len = batch
            preds = self.model(image) # shape: (seq_len, batch_size, num_classes)
            preds = preds.permute(1, 0, 2)

        pred_seqs = []

        for seq in preds:
            val = post_process(seq, self.dataset)
            pred_seqs.append(greedy_decode(val))

        target_seqs = []
        start = 0
        for end in target_len.tolist():
            tmp = target[start: start+end].tolist()
            tmp = idx2char(tmp, self.dataset)
            start = start + end
            target_seqs.append("".join(tmp))

        #scores = []
        
        #for i, j in zip(pred_seqs, target_seqs):
        #    score = char_accuracy(i, j)
        #    scores.append(score)

        #final_res = sum(scores) / len(scores)
        final_res = self.cer_metric_fn(pred_seqs, target_seqs)
        return final_res
        
    