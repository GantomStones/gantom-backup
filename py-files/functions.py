import torch
import torch.nn as nn
import numpy as np
from operator import itemgetter
from PIL import ImageOps
import json


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                   # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acce: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))



class Shape_classifier(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # in: 1 x 256 x 256
        self.network = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=2, bias=False),
                                     nn.MaxPool2d(2,2),
                                     nn.BatchNorm2d(64), 
                                     nn.LeakyReLU(0.2, inplace=True),
                                     # out: 64 x 127 x 127 

                                     nn.Conv2d(64, 128, kernel_size=4,
                                               stride=1, padding=1, bias=False),
                                     nn.MaxPool2d(2,2),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     # out: 128 x 63 x 63

                                     nn.Conv2d(128, 256, kernel_size=4,
                                               stride=1, padding=1, bias=False),
                                     nn.MaxPool2d(2,2),
                                     nn.BatchNorm2d(256),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     # out: 256 x 31 x 31
                                     
                                    nn.Conv2d(256, 256, kernel_size=4,
                                               stride=1, padding=1, bias=False),
                                     nn.MaxPool2d(2,2),
                                     nn.BatchNorm2d(256),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     # out: 256 x 15 x 15

                                     nn.Conv2d(256, 512, kernel_size=4,
                                               stride=1, padding=1, bias=False),
                                     nn.MaxPool2d(2,2),
                                     nn.BatchNorm2d(512),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     # out: 512 x 7 x 7 
                                     
                                     nn.Conv2d(512, 512, kernel_size=4,
                                               stride=1, padding=1, bias=False),
                                     nn.MaxPool2d(2,2),
                                     nn.BatchNorm2d(512),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     # out: 512 x 3 x 3 
                                     
                                     
                                     nn.Flatten(),
                                     nn.Linear(512*3*3, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 48))

    def forward(self, xb):
        return self.network(xb)



classes = ['1','10','11','11a','12','13','14','15','15a','15b','16','16a','17','17a','18','19','19a','2','20','21','22','23','24','25','26','27','28','29','3','4','4a','5','6','8','8a','9','9a','9b','U-shape','fuzz']

def getImageShape(img, model):
    img = ImageOps.grayscale(img)
    img = torch.tensor(np.array(img))
    if img.dim()==2:
        img = img[(None,)*2].float()
    _, pred = torch.max(model(img), dim=1)
    shape = classes[pred]
    return shape 


def getImageRGB(img):
    img = img.convert('RGB')
    img_arr = np.array(img).reshape(-1,3)
    pxl_rgb = [list(i) for i in img_arr if any([i[0] > 35, i[1] > 35, i[2] > 35])]
    pxl_rgb = [[round((i)/10)*10 for i in rgb] for rgb in pxl_rgb]

    x_, y_ = np.unique(pxl_rgb, axis=0, return_counts=True)
    x_ = [tuple(x_[i].tolist()) for i in range(len(x_))]
    pixl_dict = {a[0]: a[1] for a in zip(x_, y_)}

    top_5_rgb = dict(sorted(pixl_dict.items(), key=itemgetter(1), reverse=True)[:5])
    top_5 = [rgb for rgb in top_5_rgb.keys()]

    return top_5


def getCodeInfo(path, ):
    with open(path, 'r') as f:
        metadata = json.load(f)
        code = metadata['name'].split(' ')[-1]
        all_four = code[1] == code[2] == code[3] == code[4]
        return code, all_four

    