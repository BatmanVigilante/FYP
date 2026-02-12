from torch.utils.data import random_split
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from training.dataset import GlomeruliDataset
from training.loss import DiceBCELoss
from training.metrics import dice_score, iou_score, recall_score
from model.adc_res_transxnet import ADCResTransXNet

dataset = GlomeruliDataset(
    "data/patches/images",
    "data/patches/masks"
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)