import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from functions import load_data, dataset_training, cardio_cnn_resnet_18, train, validation

from quilt3 import Package
import pandas as pd

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Load data from quilt
p = Package.browse(
    "aics/integrated_transcriptomics_structural_organization_hipsc_cm", "s3://allencell"
)["actn2_pattern_ml_classifier_train"].fetch("data/")

manifest = pd.read_csv("data/metadata.csv", index_col=0)

# model save path
save_model_path = "./models/"  # save Pytorch models

# set model parameters
data_path = f"data/{manifest.DataPath[0]}"
label_lists = (
    f"data/{manifest.AnnotationDiffusePath[0]}",
    f"data/{manifest.AnnotationFibersPath[0]}",
    f"data/{manifest.AnnotationDisorganizedPunctaPath[0]}",
    f"data/{manifest.AnnotationOrganizedPunctaPath[0]}",
    f"data/{manifest.AnnotationOrganizedZDisks[0]}",
)
window_size = 96
kernel_size = 5
drop_p = 0.1

# set image augmentation transforms
tf_train = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0966], std=[0.1153]),
    ]
)

tf_val = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0966], std=[0.1153]),
    ]
)

# set training parameters
epochs = 750
batch_size = 40
learning_rate = 1e-5
log_interval = 20


# Detect devices
use_cuda = torch.cuda.is_available()  # check if GPU exists
# use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

params = (
    {"batch_size": batch_size, "shuffle": True, "num_workers": 1, "pin_memory": True}
    if use_cuda
    else {}
)
# params = {'batch_size': batch_size, 'shuffle': True}


all_X, all_y = load_data(data_path, label_lists, window_size)

train_X, test_X, train_y, test_y = train_test_split(
    all_X, all_y, test_size=0.20, random_state=42
)

# enc = OneHotEncoder(n_values=5)
# train_y = enc.fit_transform(train_y)
# test_y = enc.fit_transform(test_y)

train_set = dataset_training(train_X, train_y, transform=tf_train)
valid_set = dataset_training(test_X, test_y, transform=tf_val)

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)

cnn_model = cardio_cnn_resnet_18().to(device)
optimizer = torch.optim.SGD(
    cnn_model.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9
)

# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

# start training
for epoch in range(epochs):
    # train, test model
    train_losses, train_scores = train(
        log_interval, cnn_model, device, train_loader, optimizer, epoch
    )
    epoch_test_loss, epoch_test_score = validation(
        cnn_model, device, valid_loader, optimizer, save_model_path, epoch
    )

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_scores)
    np.save("./CNN_epoch_training_losses.npy", A)
    np.save("./CNN_epoch_training_scores.npy", B)
    np.save("./CNN_epoch_test_loss.npy", C)
    np.save("./CNN_epoch_test_score.npy", D)

# plot
fig = plt.figure(figsize=(10, 4))
plt.subplot(121)
# train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), A[:, -1])
#  test loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), C)
plt.title("model loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["train", "test"], loc="upper left")
# 2nd figure
plt.subplot(122)
# train accuracy (on epoch end)
plt.plot(np.arange(1, epochs + 1), B[:, -1])
#  test accuracy (on epoch end)
plt.plot(np.arange(1, epochs + 1), D)
plt.title("training scores")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(["train", "test"], loc="upper left")
title = "./myoModel.png"
plt.savefig(title, dpi=600)
plt.show()
