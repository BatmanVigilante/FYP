import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from training.dataset import GlomeruliDataset
from training.loss import DiceBCELoss
from training.metrics import dice_score, iou_score, recall_score
from model.adc_res_transxnet import ADCResTransXNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    print("Loading dataset...")

    dataset = GlomeruliDataset(
        "data/patches/images",
        "data/patches/masks"
    )

    print("Dataset size:", len(dataset))

    if len(dataset) == 0:
        print("ERROR: No patches found. Run patch extraction first.")
        return

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    print("Initializing model...")

    model = ADCResTransXNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = DiceBCELoss()

    num_epochs = 15

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training loop
        for imgs, masks in tqdm(train_loader):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print("Train Loss:", epoch_loss / len(train_loader))

        # Validation metrics
        model.eval()
        with torch.no_grad():
            imgs, masks = next(iter(val_loader))
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)

            print("Dice:", dice_score(preds, masks).item())
            print("IoU:", iou_score(preds, masks).item())
            print("Recall:", recall_score(preds, masks).item())

        # Save checkpoint every epoch
        torch.save(model.state_dict(), "results/adc_res_transxnet.pth")
        print("Checkpoint saved to results/adc_res_transxnet.pth")


if __name__ == "__main__":
    train()