import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.adc_res_transxnet import ADCResTransXNet
from training.dataset import GlomeruliDataset
from training.loss import DiceBCELoss
from training.metrics import dice_score, iou_score, recall_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    dataset = GlomeruliDataset(
        "data/patches/images",
        "data/patches/masks"
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    model = ADCResTransXNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = DiceBCELoss()

    for epoch in range(5):  # keep small for now
        model.train()
        epoch_loss = 0

        for imgs, masks in tqdm(loader):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"\nEpoch {epoch+1}")
        print("Loss:", epoch_loss / len(loader))

        # quick metric check
        model.eval()
        with torch.no_grad():
            imgs, masks = next(iter(loader))
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)

            print("Dice:", dice_score(preds, masks).item())
            print("IoU:", iou_score(preds, masks).item())
            print("Recall:", recall_score(preds, masks).item())


if __name__ == "__main__":
    train()