import os
import math
import numpy as np
import skimage.io as skio
from aicsimageio import AICSImage
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torchvision.models as models


def get_crop(image_set, location, window_size):
    image = image_set[location[0], :, :]
    xCoor = location[1]
    yCoor = location[2]
    imShape = np.shape(image)
    image_crop = np.zeros((window_size, window_size), dtype=np.float16)
    crop_center = math.ceil(window_size / 2)

    if window_size % 2 == 0:
        xMinus = window_size / 2
        xPlus = xMinus
        yMinus = window_size / 2
        yPlus = yMinus
    else:
        xMinus = xPlus = window_size // 2
        xMinus = yPlus = window_size // 2

    xMinus = int(np.min([xMinus, xCoor]))
    yMinus = int(np.min([yMinus, yCoor]))
    xPlus = int(np.min([xPlus, imShape[0] - xCoor]))
    yPlus = int(np.min([yPlus, imShape[1] - yCoor]))

    image_crop[
        (crop_center - xMinus) : (crop_center + xPlus),
        (crop_center - yMinus) : (crop_center + yPlus),
    ] = image[(xCoor - xMinus) : (xCoor + xPlus), (yCoor - yMinus) : (yCoor + yPlus)]
    image_crop = image_crop / np.amax(image_crop)

    return image_crop


def load_data(data_path, label_lists, window_size):

    """
    Load training data
    """

    reader = AICSImage(data_path)
    image_set = np.squeeze(reader.data)

    for i, list in enumerate(label_lists):
        loc = np.loadtxt(list, dtype=np.uint16, delimiter=" ")
        if i == 0:
            locations = loc
            labels = np.zeros((np.shape(loc)[0], 1), dtype=np.uint8)
        else:
            locations = np.concatenate((locations, loc), axis=0)
            lab = np.ones((np.shape(loc)[0], 1), dtype=np.uint8) * i
            labels = np.concatenate((labels, lab), axis=0)

    cropped_images = np.zeros((len(labels), window_size, window_size), dtype=np.float32)
    for i in range(len(labels)):
        cropped_images[i, :, :] = get_crop(image_set, locations[i, :], window_size)

    if not os.path.exists("patches/"):
        os.makedirs("patches/")

    for i in range(len(label_lists)):
        in_class = np.squeeze(labels == i)
        class_patches = cropped_images[in_class, :, :]
        skio.imsave(f"./patches/{i}_patches.tif", class_patches)

    print("Patches Saved")

    return cropped_images, labels


class dataset_training(data.Dataset):
    def __init__(self, cropped_images, labels, transform=None):
        self.image_set = []
        for i in range(len(labels)):
            if transform is not None:
                image = cropped_images[i, :, :]
                image = np.repeat(np.expand_dims(image, axis=2), repeats=3, axis=2)
                self.image_set.append(transform(image))
            else:
                image = cropped_images[i, :, :]
                image = np.repeat(np.expand_dims(image, axis=2), repeats=3, axis=2)
                self.image_set.append(image)

        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        y = torch.LongTensor(self.labels[idx])
        X = self.image_set[idx]

        return X, y


class myoCNN_resnet_18(nn.Module):
    def __init__(self, num_classes=5):
        super(myoCNN_resnet_18, self).__init__()

        resnet18 = models.resnet18(pretrained=True)
        res_modules = list(resnet18.children())[:-1]
        self.resnet18 = nn.Sequential(*res_modules)
        self.fc = self.fcClassifier(512, num_classes)

    def forward(self, x):
        x = F.interpolate(x, (224, 224))
        # x = torch.cat((x,x,x),dim=1)
        x = self.resnet18(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def fcClassifier(self, inChannels, numClasses):
        fc_classifier = nn.Sequential(
            nn.Linear(inChannels, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, numClasses),
        )

        return fc_classifier


def train(log_interval, model, device, train_loader, optimizer, epoch):

    """
    Function for CNN training
    """

    model.train()

    losses = []
    scores = []
    N_count = 0

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        N_count += X.size(0)

        optimizer.zero_grad()
        output = model(X)

        loss = F.cross_entropy(output, y.view(-1))

        losses.append(loss.item())

        y_pred = torch.max(output, 1)[1]
        step_score = accuracy_score(
            y.cpu().data.view(-1).numpy(), y_pred.cpu().data.view(-1).numpy()
        )
        scores.append(step_score)

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%".format(
                    epoch + 1,
                    N_count,
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    100 * step_score,
                )
            )

    return losses, scores


def validation(model, device, test_loader, optimizer, save_model_path, epoch):

    """
    Applying model on validation/test set
    """

    # set model as testing mode
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1,)

            output = model(X)

            if device is torch.device("cuda"):
                loss = F.cross_entropy(torch.squeeze(output), y, reduction="sum")
            else:
                loss = F.cross_entropy(output, y, reduction="sum")  # for cpu
            test_loss += loss.cpu().data  # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # to compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(
        all_y.cpu().data.view(-1).numpy(), all_y_pred.cpu().data.view(-1).numpy()
    )
    # show information
    print(
        "\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(
            len(all_y), test_loss, 100 * test_score
        )
    )

    # save Pytorch models of best record
    if not os.path.isdir(save_model_path):
        os.mkdir(save_model_path)

    torch.save(
        model.state_dict(),
        os.path.join(save_model_path, "3dcnn_epoch{}.pth".format(epoch + 1)),
    )

    return test_loss, test_score
