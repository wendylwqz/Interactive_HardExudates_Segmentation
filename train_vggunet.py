import argparse
import os
import fcntl
import math
from pathlib import Path
import random
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from automatic_module.automatic_segmentation_model import vgg16bn_unet

def arguments():
    parser = argparse.ArgumentParser(description="Train VGGUnet for EX segmentation")
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, required=True)
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "rmsprop"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.dataset_name is None):
        parser.error("Missing dataset name arg")
    if min(args.epochs, args.batch_size, args.val_every) < 1:
        parser.error("epochs, batch size, and validation interval must be positive")
    return args

def pair_image_mask_files(dataset_dir):
    images = Path(dataset_dir) / "images"
    masks = Path(dataset_dir) / "masks"

    images = sorted(path for path in images.iterdir())
    masks = sorted(path for path in masks.iterdir())

    if len(images) != len(masks):
        raise ValueError(f"expected same number of image and masks in {dataset_dir}")
    return list(zip(images, masks))

def get_ex_mask(mask, mask_path, ex_class_label):
    values = np.asarray(mask)
    if values.ndim != 2 or values.dtype.kind not in "bui":
        raise ValueError(f"expected a single channel mask with int labels in {mask_path}")
    return values == ex_class_label

def load_image_mask_pairs(pairs):
    items = []
    for img_path, mask_path in pairs:
        with Image.open(img_path) as image, Image.open(mask_path) as mask:
            target = get_ex_mask(mask, mask_path, 1)      # the EX class label =1 in our masks
            rgb_img = image.convert("RGB")
            binary_mask = Image.fromarray(target.astype(np.uint8))
        items.append((rgb_img, binary_mask))
    return items

# combines weighted binary CE & dice loss
def ex_loss(logits, mask, positive_weight):
    # logits[:, 0]   background score for every pixel in every image
    # logits[:, 1]   EX score for every pixel in every image
    log_odds = logits[:, 1].float() - logits[:, 0].float()  # how much the model favors EX over bg
    goal = mask.float()

    bce = F.binary_cross_entropy_with_logits(log_odds, goal, pos_weight=positive_weight)
    prob = log_odds.sigmoid()
    intersection = (prob * goal).sum(dim=(1, 2))
    dice_denominator = prob.sum(dim=(1, 2)) + goal.sum(dim=(1, 2))
    dice_loss = 1 - ((2 * intersection + 1) / (dice_denominator + 1)).mean()    # +1 smoothing to avoid division by 0
    return bce + dice_loss



class DatasetImages(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img, mask = self.items[index]
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255), torch.from_numpy(np.array(mask))

def save_atomic(data, dest):
    temp = dest.with_name(dest.name + ".tmp")
    torch.save(data, temp)
    temp.replace(dest)

@torch.inference_mode()
def validate(model, dataset, device):
    model.eval()
    shape = (len(dataset), 960, 1440)
    scores = np.empty(shape, dtype=np.float32)
    goals = np.empty(shape, dtype=np.int8)

    for i, (img, mask) in enumerate(dataset):
        rgb = img.unsqueeze(0).to(device)   # adds batch size to input img dimension for model inference
        scores[i] = model(rgb).softmax(dim=1)[0, 1].cpu().numpy()
        goals[i] = mask.numpy()
    avg_precision = average_precision_score(goals.ravel(), scores.ravel())
    predicted = scores >= 0.5
    dice = 2 * np.count_nonzero(predicted & goals) / (predicted.sum() + goals.sum())
    return float(avg_precision), float(dice)
    

def main():
    args = arguments()
    image_h, image_w = 960, 1440
    train_pairs = pair_image_mask_files(Path("../data") / args.dataset_name / "train")
    val_pairs = pair_image_mask_files(Path("../data") / args.dataset_name / "val")

    training = DatasetImages(load_image_mask_pairs(train_pairs))
    validation = DatasetImages(load_image_mask_pairs(val_pairs))

    output_dir = Path(args.output_dir) if args.output_dir is not None else Path("automatic_weights") / args.dataset_name / "VGGUnet_weights"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / ".training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"anther job is using {output_dir}") from None
        train(args, training, validation)

    
def train(args, training, validation):
    device = torch.device(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda is unavalable, run on gpu node or --device for trail")
    torch.manual_seed(args.seed)

    model = vgg16bn_unet(output_dim=2, pretrained=True).to(device)
    optimizer = torch.optim.SGD(model.parameter(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.9)
    best_aupr = -1.0
    best_path = args.out_dir / "VGGUNet_EX.tar"

    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, 
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    positive_weight = torch.tensor(10.0, device=device) # gives EX pixels more weight in the CE loss

    for epoch in range(1, args.epochs + 1):
        torch.manual_seed(args.seed + epoch)            # update seed num
        model.train()
        total_loss = 0.0
        for img, mask in loader:                        # train & update weights 1 batch at a time
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)       # reset pytorch gradient from prev batch
            logits = model(img)                         # forward pass 
            loss = ex_loss(logits, mask, positive_weight)#compute loss
            loss.backward()                             # compute loss gradients for trainable parameters
            optimizer.step()                            # use found gradients to update weights
            total_loss += loss.detach().item()
        print(f"epoch: {epoch}/{args.epochs};   avg loss/batch: {total_loss / len(loader):.5f};   "
              f"lr: {scheduler.get_last_lr()[0]:.6g}", flush=True)
        scheduler.step()                                # updates scheduler that an epoch has passed

        if epoch == 1 or epoch % args.val_every == 0 or epoch == args.epochs:
            aupr, dice = validate(model, validation, device)
            print(f"val aupr: {aupr:.5f};   dice: {dice:.5f}", flush=True)
            state = model.state_dict()
            if aupr > best_aupr:
                save_atomic({"state_dict": state, "validation": aupr}, best_path)
                best_aupr = aupr
                print(f"saved {best_path}", flush=True)
        



    



