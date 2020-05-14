import math
import torch
import numpy as np
from tqdm import tqdm
from skimage.transform import resize

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


class cardioCNN:
    def __init__(self, model_path):
        ##
        # Inputs:
        #   model_path: file path to .pth file containing model weights
        ##
        super(cardioCNN, self).__init__()

        self.__model = cardioCNN_model()  # resnet18 model
        self.__model.load_state_dict(torch.load(model_path))  # loading trained weights
        self.__tf_raw = transforms.Compose(
            [  # transformations for unaltered input
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0966], std=[0.1153]),
            ]
        )
        self.__tf_aug = transforms.Compose(
            [  # transformations for augmented input
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0966], std=[0.1153]),
            ]
        )
        self.__window_size = 96  # size of local patch used for classification

    def predict_image_sliding(self, raw_im, stride, batch_size, interp_order, device):
        ##
        # Inputs:
        #   raw_im: 2D numpy array of image to be classified
        #   stride: stride of sliding window to be used for classification
        #   batch_size: number of voronoi units to be evaluated concurrently
        #   inter_order: method of interpolation
        #                   0: Nearest Neighbor
        #                   1: Bi-linear
        #                   2: Bi-quadradic
        #                   3: Bi-cubic
        #                   4: Bi-quartic
        #                   5: Bi-quintic
        #   device: device for running model (cpu or cuda)
        #
        # Outputs:
        #   prob_map: 3D array (5xnxm) of per-pixel/unit class probabilities (softmax of averaged predictions)
        ##
        self.__model.to(device).eval()
        im_shape = np.shape(raw_im)
        prob_map = np.zeros(
            (5, im_shape[0] // stride, im_shape[1] // stride)
        )  # initialized probability map
        raw_im = raw_im.astype(np.float32)
        raw_im = raw_im / np.amax(raw_im)  # normalize image

        curr_batch_0 = None
        curr_batch_1 = None
        curr_batch_2 = None
        curr_batch_3 = None
        batch_r = []
        batch_c = []
        in_batch = 0

        last_r = range(0, im_shape[0], stride)[-1]
        last_c = range(0, im_shape[1], stride)[-1]

        for r in tqdm(range(0, im_shape[0], stride)):
            for c in range(0, im_shape[1], stride):

                # build input batch
                if (in_batch < (batch_size)) and (r < last_r and c < last_c):
                    patch = self.__get_patch__(raw_im, (r, c))
                    batch_r.append(r // stride)
                    batch_c.append(c // stride)

                    if in_batch == 0:
                        patch_0 = self.__tf_raw.__call__(torch.Tensor(patch)).to(device)
                        patch_1 = self.__tf_aug.__call__(torch.Tensor(patch)).to(device)
                        patch_2 = self.__tf_aug.__call__(torch.Tensor(patch)).to(device)
                        patch_3 = self.__tf_aug.__call__(torch.Tensor(patch)).to(device)

                        curr_batch_0 = patch_0[None, :, :, :]
                        curr_batch_1 = patch_1[None, :, :, :]
                        curr_batch_2 = patch_2[None, :, :, :]
                        curr_batch_3 = patch_3[None, :, :, :]
                    else:
                        patch_0 = self.__tf_raw.__call__(torch.Tensor(patch)).to(device)
                        patch_1 = self.__tf_aug.__call__(torch.Tensor(patch)).to(device)
                        patch_2 = self.__tf_aug.__call__(torch.Tensor(patch)).to(device)
                        patch_3 = self.__tf_aug.__call__(torch.Tensor(patch)).to(device)

                        curr_batch_0 = torch.cat(
                            (curr_batch_0, patch_0[None, :, :, :]), dim=0
                        )
                        curr_batch_1 = torch.cat(
                            (curr_batch_1, patch_1[None, :, :, :]), dim=0
                        )
                        curr_batch_2 = torch.cat(
                            (curr_batch_2, patch_2[None, :, :, :]), dim=0
                        )
                        curr_batch_3 = torch.cat(
                            (curr_batch_3, patch_3[None, :, :, :]), dim=0
                        )

                    in_batch += 1

                # evaluate batches with model and average predictions
                if in_batch == batch_size or (r == last_r and c == last_c):
                    predictions = (
                        self.__model(curr_batch_0).cpu().detach().numpy().squeeze() / 4
                    )
                    predictions += (
                        self.__model(curr_batch_1).cpu().detach().numpy().squeeze() / 4
                    )
                    predictions += (
                        self.__model(curr_batch_2).cpu().detach().numpy().squeeze() / 4
                    )
                    predictions += (
                        self.__model(curr_batch_3).cpu().detach().numpy().squeeze() / 4
                    )

                    # apply softmax to get class probabilities for each unit
                    predictions = (
                        F.softmax(torch.Tensor(predictions), dim=1).numpy().squeeze()
                    )

                    # apply class probabilities to corresponding units in prob_map
                    for batch_index in range(len(batch_r)):
                        br = batch_r[batch_index]
                        bc = batch_c[batch_index]
                        for bclass in range(5):
                            prob_map[bclass, br, bc] = predictions[batch_index, bclass]

                    curr_batch_0 = None
                    curr_batch_1 = None
                    curr_batch_2 = None
                    curr_batch_3 = None
                    batch_r = []
                    batch_c = []
                    in_batch = 0

        prob_map = resize(prob_map, (5, im_shape[0], im_shape[1]), order=interp_order)

        return prob_map

    def __get_patch__(self, image, location):
        ##
        # Gets cropped patch of image at given location. Images are zero-padded for edge-cases
        ##
        window_size = self.__window_size
        xCoor = int(location[0])
        yCoor = int(location[1])
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
        ] = image[
            (xCoor - xMinus) : (xCoor + xPlus), (yCoor - yMinus) : (yCoor + yPlus)
        ]
        image_crop = image_crop / np.amax(image_crop)

        return image_crop


class cardioCNN_model(nn.Module):
    def __init__(self, num_classes=5):
        super(cardioCNN_model, self).__init__()

        resnet18 = models.resnet18(pretrained=True)
        res_modules = list(resnet18.children())[:-1]
        self.resnet18 = nn.Sequential(*res_modules)
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = F.interpolate(x, (224, 224))
        x = torch.cat((x, x, x), dim=1)
        x = self.resnet18(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
