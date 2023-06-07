import random
import torch
from torch import nn
import torchvision
from pathlib import Path
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler


class TinyVGG(nn.Module):
    """Model architecture copying TinyVGG from CNN explainer"""

    def __init__(self, input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 53 * 53, out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


def walk_through_dir(dir_path):
    """Walks throuigh dir_path returining its contents. """
    for dir_path, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dir_path}")


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    model.train()
    train_loss, train_acc = 0, 0

    for (X, y) in dataloader:
        X = X.to(device)
        y = y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for (X, y) in dataloader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)

            loss = loss_fn(y_pred, y)

            test_loss += loss.item()

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            test_acc += (y_pred_class == y).sum().item() / len(y_pred)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    return test_loss, test_acc


#
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)

MODEL_NAME = "01_efficientnet_b0_model.pth"

MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Setting up device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Getting all paths
data_path = Path("/kaggle/input/100-bird-species/")
test_path = data_path / "test"
train_path = data_path / "train"
valid_path = data_path / "valid"

print(test_path, train_path, valid_path)

# walk_through_dir(train_path)


image_path_list = list(data_path.glob("*/*/*.jpg"))

random_image_path = random.choice(image_path_list)
print(random_image_path)
random_class_name = random_image_path.parent.stem
print(random_class_name)

img = Image.open(random_image_path)
img_as_array = np.asarray(img)

print(img_as_array.shape)

transforms = {"train": transforms.Compose([  # transforms.TrivialAugmentWide(num_magnitude_bins=2),
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()]),
    "test": transforms.Compose([transforms.Resize(size=(224, 224)),
                                transforms.ToTensor()]),
    "valid": transforms.Compose([transforms.Resize(size=(224, 224)),
                                 transforms.ToTensor()])}

train_data = datasets.ImageFolder(root=train_path,
                                  transform=transforms["train"],
                                  target_transform=None)

test_data = datasets.ImageFolder(root=test_path,
                                 transform=transforms["test"],
                                 target_transform=None)

valid_data = datasets.ImageFolder(root=valid_path,
                                  transform=transforms["valid"],
                                  target_transform=None)

class_names = train_data.classes
class_dict = train_data.class_to_idx

print(train_data.classes)

print(len(train_data), len(test_data), len(valid_data))

img, label = train_data[0][0], train_data[0][1]

print(img.shape, label)

BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

valid_dataloader = DataLoader(dataset=valid_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

print(next(iter(train_dataloader))[0].shape)

model_0 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(class_names))

model_1 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
# num_features = model_1.fc.in_features
# model_1.fc = nn.Linear(num_features, len(class_names))
model_1.classifier[1] = nn.Linear(in_features=1280, out_features=len(class_names))
# model_1.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
model_1 = model_1.to(device)

# train_dataloader, test_dataloader = train_dataloader.to(device), test_dataloader.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_1.parameters(), lr=0.003, momentum=0.9)
exp_lr = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# loaded_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# loaded_model.fc = nn.Linear(num_features, len(class_names))
# loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

EPOCHS = 10

best_acc = 0.0

# test_loss = test_step(model=loaded_model,
#                      dataloader=test_dataloader,
#                      loss_fn=loss_fn)


for epoch in range(EPOCHS):
    train_loss, train_acc = train_step(model=model_1,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer)
    test_loss, test_acc = test_step(model=model_1,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn)
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(obj=model_1.state_dict(),
                   f=MODEL_SAVE_PATH)
    print(
        f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | Train acc: {train_acc * 100:.2f}% | Test acc: {test_acc * 100:.2f}% ")