import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm
from aicsimageio import AICSImage
import math
from sklearn.metrics import accuracy_score, confusion_matrix
import skimage.io as skio
from math import ceil
from torch.autograd import Variable
from scipy.ndimage import zoom
from scipy.ndimage.measurements import center_of_mass
from tqdm import tqdm

## ------------------- data loading tools ------------------ ##

def load_data(data_path, label_lists, window_size):
    reader = AICSImage(data_path)
    image_set = np.squeeze(reader.data)

    for i, list in enumerate(label_lists):
        loc = np.loadtxt(list, dtype=np.uint16, delimiter=' ')
        if i == 0:
            locations = loc
            labels = np.zeros((np.shape(loc)[0],1), dtype=np.uint8)
        else:
            locations = np.concatenate((locations, loc), axis=0)
            lab = np.ones((np.shape(loc)[0], 1), dtype=np.uint8)*i
            labels = np.concatenate((labels, lab), axis=0)

    cropped_images = np.zeros((len(labels), window_size, window_size), dtype=np.float32)
    for i in range(len(labels)):
        cropped_images[i,:,:] = get_crop(image_set, locations[i,:], window_size)

    if not os.path.exists('.patches/'):
        os.makedirs('patches/')

    for i in range(len(label_lists)):
        in_class = np.squeeze(labels==i)
        class_patches = cropped_images[in_class,:,:]
        skio.imsave(f'./patches/{i}_patches.tif', class_patches)

    #print('Mean: {}'.format(np.mean(cropped_images.flatten())))
    #print('StDev: {}'.format(np.std(cropped_images.flatten())))
    
    print('Patches Saved')

    return cropped_images, labels

def eval_image(image, model, window_size):
    model.eval()

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0966], std=[0.1153])
    ])

    im_shape = np.shape(image)
    nr = im_shape[0]
    nc = im_shape[1]
    prob_map = np.zeros((nr, nc, 5))

    for r in range(nr):
        for c in range(nc):
            patch = get_patch(image, (r,c), window_size)
            patch = tf.__call__(torch.Tensor(patch))
            patch = patch[None,:,:,:]

            pred = F.softmax(model(patch), dim=-1).detach().numpy()
            prob_map[r,c,:] = np.squeeze(pred)

    return prob_map

def eval_image_subsample(image, model, window_size, stride, device):
    model.eval()

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0966], std=[0.1153])
    ])

    im_shape = np.shape(image)
    nr = im_shape[0]//stride
    nc = im_shape[1]//stride
    prob_map = np.zeros((5, nr, nc))

    for r in tqdm(range(nr)):
        for c in range(nc):
            patch = get_patch(image, (int(r*stride),int(c*stride)), window_size)
            patch = tf.__call__(torch.Tensor(patch)).to(device)
            patch = patch[None,:,:,:]

            pred = F.softmax(model(patch), dim=-1).cpu().detach().numpy()
            prob_map[:,r,c] = np.squeeze(pred)

    prob_map = zoom(prob_map, (1, im_shape[0]/nr, im_shape[1]/nc), order=0)

    return prob_map

def eval_image_voronoi(image, voronoi, model, window_size, device):
    model.to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0966], std=[0.1153])
    ])

    im_shape = np.shape(image)
    prob_map = np.zeros((5, im_shape[0], im_shape[1]))

    numUnits = np.amax(voronoi)
    unit = np.zeros_like(voronoi)
    for u in tqdm(range(numUnits)):
        unit[:] = 0
        unit[voronoi==u] = 1

        center = center_of_mass(unit)
        patch = get_patch(image, (int(center[0]), int(center[1])), window_size)
        patch = tf.__call__(torch.Tensor(patch)).to(device)
        patch = patch[None,:,:,:]

        pred = np.squeeze(F.softmax(model(patch), dim=-1).cpu().detach().numpy())

        for c in range(5):
            prob_map[c][voronoi==u] = pred[c]

    return prob_map


def get_patch(image, location, window_size):
    xCoor = location[0]
    yCoor = location[1]
    imShape = np.shape(image)
    image_crop = np.zeros((window_size,window_size),dtype=np.float16)
    crop_center = math.ceil(window_size/2)

    if window_size % 2 == 0:
        xMinus = window_size/2
        xPlus = xMinus
        yMinus = window_size/2
        yPlus = yMinus
    else:
        xMinus = xPlus = window_size//2
        xMinus = yPlus = window_size//2

    xMinus = int(np.min([xMinus, xCoor]))
    yMinus = int(np.min([yMinus, yCoor]))
    xPlus = int(np.min([xPlus, imShape[0]-xCoor]))
    yPlus = int(np.min([yPlus, imShape[1]-yCoor]))

    image_crop[(crop_center-xMinus):(crop_center+xPlus), (crop_center-yMinus):(crop_center+yPlus)] = image[(xCoor-xMinus):(xCoor+xPlus), (yCoor-yMinus):(yCoor+yPlus)]
    image_crop = image_crop/np.amax(image_crop)

    return image_crop


def get_crop(image_set, location, window_size):
    image = image_set[location[0],:,:]
    xCoor = location[1]
    yCoor = location[2]
    imShape = np.shape(image)
    image_crop = np.zeros((window_size,window_size),dtype=np.float16)
    crop_center = math.ceil(window_size/2)

    if window_size % 2 == 0:
        xMinus = window_size/2
        xPlus = xMinus
        yMinus = window_size/2
        yPlus = yMinus
    else:
        xMinus = xPlus = window_size//2
        xMinus = yPlus = window_size//2

    xMinus = int(np.min([xMinus, xCoor]))
    yMinus = int(np.min([yMinus, yCoor]))
    xPlus = int(np.min([xPlus, imShape[0]-xCoor]))
    yPlus = int(np.min([yPlus, imShape[1]-yCoor]))

    image_crop[(crop_center-xMinus):(crop_center+xPlus), (crop_center-yMinus):(crop_center+yPlus)] = image[(xCoor-xMinus):(xCoor+xPlus), (yCoor-yMinus):(yCoor+yPlus)]
    image_crop = image_crop/np.amax(image_crop)

    return image_crop

## ---------------------- Dataloader ---------------------- ##
class Dataset_Training(data.Dataset):
    def __init__(self, cropped_images, labels, transform=None):
        self.image_set = []
        for i in range(len(labels)):
            if transform is not None:
                self.image_set.append(transform(cropped_images[i,:,:]))
            else:
                self.image_set.append(cropped_images[i,:,:])

        self.labels = labels # F.one_hot(torch.from_numpy(labels).long(), num_classes=5)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        y = torch.LongTensor(self.labels[idx])
        X = self.image_set[idx]

        return X, y


## ------------------------ CNN module ---------------------- ##
class myoCNN(nn.Module):
    def __init__(self, window_size=128, drop_p=0.2, kernel=5, num_classes=5, batch_size=10):
        super(myoCNN,self).__init__()

        self.drop_p = drop_p
        self.poolSize = 2
        self.kernel_size = kernel
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.cb1 = 128
        self.cb2 = 128
        self.cb3 = 64
        self.flattened = (window_size//8)**2*self.cb3 #2592 #
        self.pad = kernel//2


        self.convBlock1 = self.convBlock(1, self.cb1, self.kernel_size)
        self.convBlock2 = self.convBlock(self.cb1, self.cb2, self.kernel_size)
        self.convBlock3 = self.convBlock(self.cb2, self.cb3, self.kernel_size)
        self.pool = nn.MaxPool2d(self.poolSize)
        self.classifier = self.fcClassifier(self.flattened, self.num_classes)


    def forward(self,x):
        #import pdb; pdb.set_trace()
        x = self.convBlock1(x)
        x = self.pool(x)

        x = self.convBlock2(x)
        x = self.pool(x)

        x = self.convBlock3(x)
        x = self.pool(x)

        x = x.view(self.batch_size, -1)

        x = self.classifier(x)

        return x



    def convBlock(self, inChannels, outChannels, kernelSize):
        conv_block = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=kernelSize, stride=1, padding=self.pad),
            nn.BatchNorm2d(outChannels),
            nn.Conv2d(outChannels, outChannels, kernel_size=kernelSize, stride=1, padding=self.pad),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(),
            nn.Dropout(self.drop_p)
        )

        return conv_block

    def fcClassifier(self, inChannels, numClasses):
        fc_classifier = nn.Sequential(
            nn.Linear(inChannels, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 64),
            nn.Linear(64, numClasses)
        )

        return fc_classifier

class myoCNN_multiscale(nn.Module):
    def __init__(self, window_size=128, drop_p=0.2, kernel=5, num_classes=5, batch_size=10):
        super(myoCNN_multiscale,self).__init__()

        self.drop_p = drop_p
        self.poolSize = 2
        self.kernel_size = kernel
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.window_size = window_size

        self.cb1 = 128
        self.cb2 = 64
        self.cb3 = 32
        self.flattened = (window_size//8)**2*self.cb3 #2592 #
        self.pad = kernel//2


        self.inputBlock1 = self.inputBlock(self.cb1, self.kernel_size)
        self.inputBlock2 = self.inputBlock(self.cb1, self.kernel_size)
        self.inputBlock3 = self.inputBlock(self.cb2, 3)

        self.convBlock1 = self.convBlock(self.cb1, self.cb1, self.kernel_size)
        self.convBlock2 = self.convBlock(self.cb1, self.cb2, self.kernel_size)
        self.convBlock3 = self.convBlock(self.cb2, self.cb3, self.kernel_size)
        self.pool = nn.MaxPool2d(self.poolSize)
        self.classifier = self.fcClassifier(self.flattened, self.num_classes)

    def centerCropTensor(self, x, cropSize):

        sh = np.array(x.shape)

        m = p = cropSize//2
        c = ceil(sh[-1]/2)

        return x[:,:, c-m:c+p, c-m:c+p]

    def forward(self,x):
        x1 = self.inputBlock1(x)
        x2 = self.inputBlock2(self.centerCropTensor(x, self.window_size//3*2))
        x3 = self.inputBlock3(self.centerCropTensor(x, self.window_size//3))

        x = self.convBlock1(x1)
        x = self.pool(x)
        x += F.interpolate(x2, (x.shape[2], x.shape[3]))
        

        x = self.convBlock2(x)
        x = self.pool(x)
        x += F.interpolate(x3, (x.shape[2], x.shape[3]))

        x = self.convBlock3(x)
        x = self.pool(x)

        #import pdb; pdb.set_trace()

        x = x.view(x.shape[0], -1)

        x = self.classifier(x)

        return x

    def inputBlock(self, outChannels, kernelSize):
        input_block = nn.Sequential(
            nn.Conv2d(1, outChannels, kernel_size=kernelSize, stride=1, padding=kernelSize//2),
            nn.BatchNorm2d(outChannels),
            nn.Conv2d(outChannels, outChannels, kernel_size=kernelSize, stride=1, padding=kernelSize//2),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(),
            nn.Dropout(self.drop_p)
        )

        return input_block

    def convBlock(self, inChannels, outChannels, kernelSize):
        conv_block = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=kernelSize, stride=1, padding=self.pad),
            nn.BatchNorm2d(outChannels),
            nn.Conv2d(outChannels, outChannels, kernel_size=kernelSize, stride=1, padding=self.pad),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(),
            nn.Dropout(self.drop_p)
        )

        return conv_block

    def fcClassifier(self, inChannels, numClasses):
        fc_classifier = nn.Sequential(
            nn.Linear(inChannels, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 64),
            nn.Linear(64, numClasses)
        )

        return fc_classifier

class myoCNN_inception(nn.Module):
    def __init__(self, window_size=128, drop_p=0.2, kernel=5, num_classes=5, batch_size=10):
        super(myoCNN_multiscale,self).__init__()

        self.drop_p = drop_p
        self.poolSize = 2
        self.kernel_size = kernel
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.window_size = window_size

        self.cb1 = 128
        self.cb2 = 64
        self.cb3 = 32
        self.flattened = (window_size//8)**2*self.cb3 #2592 #
        self.pad = kernel//2


        self.inputBlock1 = self.inputBlock(self.cb1, self.kernel_size)
        self.inputBlock2 = self.inputBlock(self.cb1, self.kernel_size)
        self.inputBlock3 = self.inputBlock(self.cb2, 3)

        self.convBlock1 = self.convBlock(self.cb1, self.cb1, self.kernel_size)
        self.convBlock2 = self.convBlock(self.cb1, self.cb2, self.kernel_size)
        self.convBlock3 = self.convBlock(self.cb2, self.cb3, self.kernel_size)
        self.pool = nn.MaxPool2d(self.poolSize)
        self.classifier = self.fcClassifier(self.flattened, self.num_classes)

    def centerCropTensor(self, x, cropSize):

        sh = np.array(x.shape)

        m = p = cropSize//2
        c = ceil(sh[-1]/2)

        return x[:,:, c-m:c+p, c-m:c+p]

    def forward(self,x):
        x1 = self.inputBlock1(x)
        x2 = self.inputBlock2(self.centerCropTensor(x, self.window_size//3*2))
        x3 = self.inputBlock3(self.centerCropTensor(x, self.window_size//3))

        x = self.convBlock1(x1)
        x = self.pool(x)
        x += F.interpolate(x2, (x.shape[2], x.shape[3]))
        

        x = self.convBlock2(x)
        x = self.pool(x)
        x += F.interpolate(x3, (x.shape[2], x.shape[3]))

        x = self.convBlock3(x)
        x = self.pool(x)

        #import pdb; pdb.set_trace()

        x = x.view(x.shape[0], -1)

        x = self.classifier(x)

        return x

    def inputBlock(self, outChannels, kernelSize):
        input_block = nn.Sequential(
            nn.Conv2d(1, outChannels, kernel_size=kernelSize, stride=1, padding=kernelSize//2),
            nn.BatchNorm2d(outChannels),
            nn.Conv2d(outChannels, outChannels, kernel_size=kernelSize, stride=1, padding=kernelSize//2),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(),
            nn.Dropout(self.drop_p)
        )

        return input_block

    def fcClassifier(self, inChannels, numClasses):
        fc_classifier = nn.Sequential(
            nn.Linear(inChannels, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 64),
            nn.Linear(64, numClasses)
        )

        return fc_classifier

class myoCNN_ResNet_18(nn.Module):
    def __init__(self, num_classes=5):
        super(myoCNN_ResNet_18, self).__init__()

        resnet18 = models.resnet18(pretrained=True)
        res_modules = list(resnet18.children())[:-1]
        #l = 0
        self.resnet18 = nn.Sequential(*res_modules)
        #for p in self.resnet18.parameters():
        #    if l > 0:
        #    p.requires_grad = False
            #   l += 1
        self.fc = self.fcClassifier(512,num_classes)

    def forward(self, x):
        x = F.interpolate(x, (224,224))
        x = torch.cat((x,x,x),dim=1)
        x = self.resnet18(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def fcClassifier(self, inChannels, numClasses):
        fc_classifier = nn.Sequential(
            nn.Linear(inChannels, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, numClasses)
        )

        return fc_classifier

class myoCNN_ResNet_34(nn.Module):
    def __init__(self, num_classes=5):
        super(myoCNN_ResNet_34, self).__init__()

        resnet34 = models.resnet34(pretrained=True)
        res_modules = list(resnet34.children())[:-1]
        l = 0
        self.resnet34 = nn.Sequential(*res_modules)
        for p in self.resnet34.parameters():
            if l > 0:
                p.requires_grad = False
                l += 1
        self.fc = self.fcClassifier(512,num_classes)

    def forward(self, x):
        x = F.interpolate(x, (224,224))
        x = torch.cat((x,x,x),dim=1)
        #import pdb; pdb.set_trace()
        x = self.resnet34(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def fcClassifier(self, inChannels, numClasses):
        fc_classifier = nn.Sequential(
            nn.Linear(inChannels, numClasses)
        )

        return fc_classifier

## -------------------- inception modules ---------------------- ##

class Inception_Block(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(Inception_Block, self).__init__()
        self.branch1x1 = nn.BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = nn.BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = nn.BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = nn.BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = nn.BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = nn.BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

        
## -------------------- run model functions ---------------------- ##

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()

    losses = []
    scores = []
    N_count = 0

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        N_count += X.size(0)

        optimizer.zero_grad()
        output = model(X)

        #print(output.shape)
        #print(X.shape)
        #print(y.shape)
        loss = F.cross_entropy(output, y.view(-1))

        losses.append(loss.item())


        y_pred = torch.max(output,1)[1]
        step_score = accuracy_score(y.cpu().data.view(-1).numpy(), y_pred.cpu().data.view(-1).numpy())
        scores.append(step_score)

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses, scores

def validation(model, device, test_loader, optimizer, save_model_path, epoch):
    # set model as testing mode
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = model(X)

            loss = F.cross_entropy(torch.squeeze(output), y, reduction='sum')
            test_loss += loss.cpu().data                # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # to compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.view(-1).numpy(), all_y_pred.cpu().data.view(-1).numpy())
    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record
    torch.save(model.state_dict(), os.path.join(save_model_path, '3dcnn_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, '3dcnn_optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score

def myoCNN_final_prediction(model, device, loader):
    model.eval()

    all_y_probs = []
    all_y_pred = []
    all_y = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            X = X.to(device)
            output = model(X)
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            y_probs = F.softmax(output)
            #import pdb; pdb.set_trace()
            all_y_probs.append(y_probs.cpu().data.squeeze().numpy())
            all_y_pred.append(y_pred.cpu().data.squeeze().numpy())
            all_y.append(y.cpu().data.squeeze().numpy())

    
    confusion_mat = confusion_matrix(all_y, all_y_pred)

    return all_y, all_y_pred, all_y_probs, confusion_mat