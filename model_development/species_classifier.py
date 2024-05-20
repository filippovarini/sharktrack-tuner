#%%

import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from lightning.fabric import Fabric, seed_everything
from torchmetrics.classification import Accuracy

def scan_dataset(image_classification_folder: Path):
    label_names = [f.name for f in image_classification_folder.glob("*")]
    label_ids = []
    imagefiles = []
    for imagefile in image_classification_folder.rglob("*.jpg"):
        label_ids.append(label_names.index(imagefile.parent.name))
        imagefiles.append(str(imagefile))
        
    print(f"Training classifier for classes: {label_names}")
    assert len(imagefiles) == len(label_ids), f"{len(imagefiles)=} different from {len(label_ids)=}"
    return imagefiles, label_ids, label_names


def initialize_model(num_classes):
    model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.features.denseblock3.parameters():
        param.requires_grad = True

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    return model


def get_training_configurations():
    normalise_intensity = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_configs = {
        "batch_size": 42,
        "epochs": 200,
        "patience": 20,
        "kfolds": 5,
        "seed": 42,
        "save_model": False,
        "dry_run": True,
        "transformations": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            normalise_intensity]
        )
    }

    return train_configs


def train_step(model, data_loader, optimizer, fabric, epoch, **train_params):
    print(f"Training epoch {epoch}")
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)
        _, pred = torch.max(output, 1)
        loss = F.cross_entropy(pred, target)
        fabric.backward(loss)

        optimizer.step()

        if train_params["dry_run"]:
            break


def validate_step(model, data_loader, fabric, fold, acc_metric, **train_config):
    model.eval()
    loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            _, pred = torch.max(output, 1)
            loss += F.cross_entropy(pred, target)

            # Accuracy with torchmetrics
            acc_metric.update(output, target)

            if train_config["dry_run"]:
                break

    # all_gather is used to aggregate the value across processes
    loss = fabric.all_gather(loss).sum() / len(data_loader.dataset)

    # compute acc
    acc = acc_metric.compute() * 100

    print(f"\nFor fold: {fold} Validation set: Average loss: {loss:.4f}, Accuracy: ({acc:.0f}%)\n")
    return acc

def run(image_classification_folder: Path):
    train_configs = get_training_configurations()
    
    fabric = Fabric()
    seed_everything(train_configs["seed"], workers=True)

    dataset = datasets.ImageFolder(image_classification_folder, train_configs["transformations"])
    images, label_ids, label_names = scan_dataset(image_classification_folder)

    kfold = StratifiedKFold(train_configs["kfolds"], shuffle=True, random_state=train_configs["seed"])
    splits = kfold.split(images, label_ids)

    models = [initialize_model(len(label_names)) for _ in range(kfold.n_splits)]
    optimizers = [optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9) for model in models]

    for i in range(kfold.n_splits):
        models[i], optimizers[i] = fabric.setup(models[i], optimizers[i])

    acc_metric = Accuracy(task="multiclass", num_classes=10).to(fabric.device)

    for epoch in range(1, train_configs["epochs"] + 1):
        epoch_acc = 0
        for fold, (train_ids, val_ids) in enumerate(splits):
            print(f"Working on fold {fold}")

            train_dataset = Subset(dataset, train_ids)
            val_dataset = Subset(dataset, val_ids)
            train_loader = DataLoader(train_dataset, batch_size=train_configs["batch_size"], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=train_configs["batch_size"], shuffle=True)

            # set up dataloaders to move data to the correct device
            train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

            fold_model, optimizer = models[fold], optimizers[fold]

            # train and validate
            train_step(fold_model, train_loader, optimizer, fabric, epoch, **train_configs)
            epoch_acc = validate_step(fold_model, val_loader, fabric, fold, acc_metric, **train_configs)
            acc_metric.reset()

        # log epoch metrics
        print(f"Epoch {epoch} - Average acc: {epoch_acc / kfold.n_splits}")

        if train_configs["dry_run"]:
            break

    if train_configs["save_model"]:
        fabric.save(model.state_dict(), "model.pt")

run(Path("/vol/biomedic3/bglocker/ugproj2324/fv220/dev/sharktrack-tuner/data/development/v35data/image_classification"), )
# %%
