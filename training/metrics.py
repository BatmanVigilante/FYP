import torch

def dice_score(preds, targets, smooth=1e-6):
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    return (2 * intersection + smooth) / (
        preds.sum() + targets.sum() + smooth
    )


def iou_score(preds, targets, smooth=1e-6):
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def recall_score(preds, targets, smooth=1e-6):
    preds = (preds > 0.5).float()
    true_positive = (preds * targets).sum()
    possible_positive = targets.sum()
    return (true_positive + smooth) / (possible_positive + smooth)