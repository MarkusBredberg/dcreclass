#!/usr/bin/env python3
import argparse
import os
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ─────────────────────────────────────────────────────────────────────────────
# 1) Your RustigeClassifier (with num_classes=2)
# ─────────────────────────────────────────────────────────────────────────────
class RustigeClassifier(nn.Module):
    def __init__(self, input_shape, num_classes=2):
        super().__init__()
        C, H, W = input_shape
        self.conv1 = nn.Conv2d(C,  8, kernel_size=3, stride=2, padding=1)
        self.ln1   = nn.LayerNorm([8, H//2,  W//2])
        self.act1  = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.ln2   = nn.LayerNorm([16, H//4,  W//4])
        self.act2  = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(16,32, kernel_size=3, stride=2, padding=1)
        self.ln3   = nn.LayerNorm([32, H//8,  W//8])
        self.act3  = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(32,16, kernel_size=3, stride=2, padding=1)
        self.ln4   = nn.LayerNorm([16, H//16, W//16])
        self.act4  = nn.LeakyReLU()
        self.conv5 = nn.Conv2d(16,16, kernel_size=2, stride=2)
        self.ln5   = nn.LayerNorm([16, H//32, W//32])
        self.act5  = nn.LeakyReLU()
        flat = 16 * (H//32) * (W//32)
        self.fc1   = nn.Linear(flat, 100)
        self.act6  = nn.LeakyReLU()
        self.fc2   = nn.Linear(100, num_classes)
        self.sm    = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.act1(self.ln1(self.conv1(x)))
        x = self.act2(self.ln2(self.conv2(x)))
        x = self.act3(self.ln3(self.conv3(x)))
        x = self.act4(self.ln4(self.conv4(x)))
        x = self.act5(self.ln5(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = self.act6(self.fc1(x))
        x = self.fc2(x)
        return self.sm(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2) Training/eval loop
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    all_preds, all_labels = [], []
    running_loss = 0.0
    for imgs, labs in loader:
        imgs, labs = imgs.to(device), labs.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, labs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        all_preds.append(preds.argmax(1).cpu())
        all_labels.append(labs.cpu())
    preds = torch.cat(all_preds)
    labs  = torch.cat(all_labels)
    acc   = accuracy_score(labs, preds)
    return running_loss/len(loader.dataset), acc

def eval_epoch(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    running_loss = 0.0
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            preds = model(imgs)
            loss  = criterion(preds, labs)
            running_loss += loss.item() * imgs.size(0)
            all_preds.append(preds.argmax(1).cpu())
            all_labels.append(labs.cpu())
    preds = torch.cat(all_preds)
    labs  = torch.cat(all_labels)
    acc   = accuracy_score(labs, preds)
    return running_loss/len(loader.dataset), acc, labs, preds


# ─────────────────────────────────────────────────────────────────────────────
# 3) Main: parse args, build data, run
# ─────────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   type=str, required=True,
                   help="should contain `real/` and `fake/` subfolders")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--device",     type=str, choices=["cpu","cuda"], default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # transforms: resize to 128×128, to tensor, normalize [0,1]
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128,128)),
        transforms.ToTensor(),
    ])

    # Use ImageFolder: real=class0, fake=class1
    ds = datasets.ImageFolder(root=args.data_dir, transform=tf)
    assert set(ds.class_to_idx.keys()) == {"fake","real"}  or set(ds.class_to_idx.keys())=={"real","fake"}
    # remap so that real→0, fake→1
    ds.class_to_idx = {"real":0,"fake":1}
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # split 80/20
    n = len(ds)
    n_train = int(n*0.8)
    train_ds, val_ds = torch.utils.data.random_split(ds,[n_train,n-n_train])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,num_workers=4)

    model = RustigeClassifier(input_shape=(1,128,128), num_classes=2).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for ep in range(1, args.epochs+1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc,  y_true,   y_pred  = eval_epoch(model, val_loader, criterion, device)

        print(f"Epoch {ep:02d}  "
              f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}  "
              f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "rustige_real_vs_fake.pth")

    # final evaluation on the validation split
    _, _, y_true, y_pred = eval_epoch(model, val_loader, criterion, device)
    print("\n=== FINAL EVAL ON VAL SPLIT ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["real","fake"]))
