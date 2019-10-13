import os
import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torchsummary import summary


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def read_data(img_path, mode="RGB"):
    imgs_name = os.listdir(img_path)
    imgs_dir = [os.path.join(img_path, name) for name in imgs_name]
    imgs = []
    for path in imgs_dir:
        if mode == "L":
            img = cv2.imread(path, 0)
            img = cv2.resize(img, (512, 512))
            # img = img.reshape(512, 512, 1)
        else:
            img = cv2.imread(path)
            img = cv2.resize(img, (512, 512))
            # img = img.reshape(512, 512, 3)

        imgs.append(img)
    return imgs


class My_Dataset(Dataset):
    def __init__(self, img_path, label_path, transform=None):
        self.imgs = read_data(img_path)
        self.labels = read_data(label_path, mode="L")
        self.tranform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        if self.tranform:
            img = self.tranform(img)
            label = self.tranform(label)
        return [img, label]


trans = transforms.Compose([
    transforms.ToTensor(),
])

train_set = My_Dataset('../data/DRIVE/training/images', "../data/DRIVE/training/label", transform=trans)
val_set = My_Dataset('../data/DRIVE/test/images', "../data/DRIVE/test/label", transform=trans)

batch_size = 2
data_loader = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
}


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out


model = UNet_model().to(device)
summary(model, input_size=(3, 512, 512))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# train
for epoch in range(60):
    for i, (images, labels) in enumerate(data_loader['train']):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch[{}], Step[{}], loss: {:.4f}'.format(epoch, i, loss.item()))

model.eval()
torch.save(model.state_dict(), 'UNet.ckpt')

# test
for i, (images, labels) in enumerate(data_loader['val']):
    images = images.to(device)
    labels = labels.to(device)
    output = model(images)

    img = output[0][0]
    torchvision.utils.save_image(img, str(i) + ".png")


