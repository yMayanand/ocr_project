import gc
import logging
import math
import time
import os
from argparse import ArgumentParser

import torch
from model import CRNN
from module import Engine
from utils import *

# Create and configure logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

LOGGER = logging.getLogger('OCRRocket')


class Rocket:
    def __init__(self, engine, args):
        self.ckpt_dir = args.ckpt_dir
        self.engine = engine
        self.epochs = args.epochs
        self.start_epoch = 0
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.loss_meter = AverageMeter()
        self.last_epoch = 0

    def launch(self):
        LOGGER.info('🚀 Rocket Launched')

        self.engine.model.to(self.device)

        # look if resume_path specified
        if self.args.resume_path is not None:
            self.launch_resume_routine()
    
        start = self.start_epoch
        best_score =  0
        print(self.engine.optimizer.param_groups[0]['lr'])
        for epoch in range(start, start+self.epochs):
            self.engine.model.train()
            for batch_idx, batch in enumerate(self.engine.train_dl):
                batch = batch_to_device(batch, self.device)
                loss = self.engine.train_step(batch)
                loss.backward()
                # clip here
                #torch.nn.utils.clip_grad_norm_(self.engine.model.parameters(), 1.0) # or some other value

                self.engine.optimizer.step()
                self.engine.optimizer.zero_grad()
                
                self.loss_meter.update(loss.item())
            msg = colorstr(f"EPOCH{epoch} LOSS:- ") \
                + colorstr('magenta', 'bold', f"{self.loss_meter.avg:.3f}")
            self.loss_meter.reset()
            self.last_epoch = epoch
            LOGGER.info(msg)

            # Validation
            val_metric = []
            for batch_idx, batch in enumerate(self.engine.val_dl):
                batch = batch_to_device(batch, self.device)
                metric = self.engine.val_step(batch)
                val_metric.append(metric)
            avg_metric = sum(val_metric) / len(val_metric)

            msg = colorstr(f"EPOCH{epoch} CER:- ") \
                + colorstr('green', 'bold', f"{avg_metric:.3f}\n")
            LOGGER.info(msg)

            if avg_metric > best_score:
                best_score = avg_metric
                time_stats = list(time.localtime())[:5]
                time_stats = map(str, time_stats)
                filename = (
                    "best" + "_".join(time_stats) 
                    + f"_{best_score}.pt")

                self.save(filename)

        gc.collect()
        torch.cuda.empty_cache()
        
        self.save()
        
    def save(self, filename='ckpt.pt'):
        LOGGER.info("🧳 saving model...")
        save_dct = {
            "weights": self.engine.model.state_dict(),
            "optimizer_state": self.engine.optimizer.state_dict(),
            "last_epoch": self.last_epoch
        }

        torch.save(save_dct, os.path.join(self.ckpt_dir, filename))

    def launch_resume_routine(self):
        LOGGER.info("🪄 resuming training...")
        path = self.args.resume_path
        assert os.path.exists(path), "resume_path: {path} does not exists"
        state_dict = torch.load(path)
        self.engine.model.load_state_dict(state_dict['weights'])
        self.engine.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.start_epoch = state_dict['last_epoch']
        self.engine.optimizer.param_groups[0]['lr'] = self.args.lr


parser = ArgumentParser()
parser.add_argument(
    '--root_dir',
    type=str,
    help="directory where data is stored"
)

parser.add_argument(
    '--epochs',
    type=int,
    help="number of epochs to train"
)

parser.add_argument(
    '--ckpt_dir',
    type=str,
    help="directory where model will be stored"
)

parser.add_argument(
    '--lr',
    type=float,
    help="learning rate"
)

parser.add_argument(
    '--wd',
    type=float,
    help="weight decay"
)

parser.add_argument(
    '--resume_path',
    default=None,
    type=str,
    help="path of checkpoint from where we want to resume training"
)

args = parser.parse_args()
model = CRNN(80, 1, 37, 512)
engine = Engine(model, args)
rocket = Rocket(engine, args)
rocket.launch()