import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import StratifiedKFold
import numpy as np
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define data transformations
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}


def get_dataloaders(data_dir, batch_size=32):
    dataset = datasets.ImageFolder(data_dir)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    targets = [s[1] for s in dataset.samples]
    splits = list(skf.split(dataset.samples, targets))

    dataloaders = []
    for fold, (train_ids, val_ids) in enumerate(splits):
        train_dataset = Subset(
            datasets.ImageFolder(data_dir, transform=data_transforms["train"]),
            train_ids,
        )
        val_dataset = Subset(
            datasets.ImageFolder(data_dir, transform=data_transforms["val"]), val_ids
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )

        dataloaders.append({"train": train_loader, "val": val_loader})

    class_names = dataset.classes
    return dataloaders, class_names


def initialize_model(num_classes):
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features.denseblock3.parameters():
        param.requires_grad = True
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    return model


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        self.val_loss_min = val_loss
        torch.save(model.state_dict(), "checkpoint.pt")


def train_model(
    model, criterion, optimizer, scheduler, dataloaders, num_epochs=25, patience=7
):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            if phase == "val":
                early_stopping(epoch_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    model.load_state_dict(torch.load("checkpoint.pt"))
                    return model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model


def main(data_dir, num_epochs=25, patience=7, batch_size=32):
    dataloaders, class_names = get_dataloaders(data_dir, batch_size)
    all_fold_accuracies = []

    for fold, loaders in enumerate(dataloaders):
        print(f"Fold {fold}/{len(dataloaders) - 1}")
        print("-" * 10)

        model = initialize_model(num_classes=len(class_names))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.001,
            momentum=0.9,
        )
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        model = train_model(
            model,
            criterion,
            optimizer,
            exp_lr_scheduler,
            loaders,
            num_epochs=num_epochs,
            patience=patience,
        )

        model.eval()
        running_corrects = 0
        for inputs, labels in loaders["val"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        fold_acc = running_corrects.double() / len(loaders["val"].dataset)
        print(f"Fold {fold} validation accuracy: {fold_acc:.4f}")
        all_fold_accuracies.append(fold_acc)

    average_accuracy = sum(all_fold_accuracies) / len(all_fold_accuracies)
    print(f"Average validation accuracy across folds: {average_accuracy:.4f}")
    torch.save(model.state_dict(), "densenet_model.pt")
    print("Model saved successfully.")


if __name__ == "__main__":
    data_dir = "/content/image_classification1"  # Adjust the path as needed
    main(data_dir)
