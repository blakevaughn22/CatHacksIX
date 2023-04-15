import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
import os, time

class ColorizationNet(nn.Module):
  def __init__(self, input_size=128):
    super(ColorizationNet, self).__init__()
    MIDLEVEL_FEATURE_SIZE = 128

    # ResNet
    resnet = models.resnet18(num_classes=365) 
    # Change first conv layer to accept single-channel (grayscale) input
    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    # Extract midlevel features from ResNet-gray
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

    # CNN
    self.upsample = nn.Sequential(     
      nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
    )

  def forward(self, input):

    # Pass input through ResNet-gray to extract features
    midlevel_features = self.midlevel_resnet(input)

    # Upsample to get colors
    output = self.upsample(midlevel_features)
    return output

class GrayscaleImageFolder(datasets.ImageFolder):
  def __getitem__(self, index):
    path, target = self.imgs[index]
    img = self.loader(path)
    if self.transform is not None:
      img_original = self.transform(img)
      img_original = np.asarray(img_original)
      img_lab = rgb2lab(img_original)
      img_lab = (img_lab + 128) / 255
      img_ab = img_lab[:, :, 1:3]
      img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
      img_original = rgb2gray(img_original)
      img_original = torch.from_numpy(img_original).unsqueeze(0).float()
    if self.target_transform is not None:
      target = self.target_transform(target)
    return img_original, img_ab, target

class AverageMeter(object):
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
  plt.clf()
  color_image = torch.cat((grayscale_input, ab_input), 0).numpy()
  color_image = color_image.transpose((1, 2, 0))
  color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
  color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
  color_image = lab2rgb(color_image.astype(np.float64))
  grayscale_input = grayscale_input.squeeze().numpy()
  if save_path is not None and save_name is not None: 
    plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
    plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))

def validate(val_loader, model, criterion, save_images, epoch):
  model.eval()

  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

  end = time.time()
  for i, (input_gray, input_ab, target) in enumerate(val_loader):
    data_time.update(time.time() - end)

    if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

    output_ab = model(input_gray) 
    loss = criterion(output_ab, input_ab)
    losses.update(loss.item(), input_gray.size(0))

    batch_time.update(time.time() - end)
    end = time.time()

    # Print MSE
    print('Validate: [{0}/{1}]\t'
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            i+1, len(val_loader), batch_time=batch_time, loss=losses))

  print('Finished validation.')
  return losses.avg

def train(train_loader, model, criterion, optimizer, epoch):
  print('Starting training epoch {}'.format(epoch))
  model.train()
  
  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

  end = time.time()
  for i, (input_gray, input_ab, target) in enumerate(train_loader):
    
    if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

    data_time.update(time.time() - end)

    output_ab = model(input_gray) 
    loss = criterion(output_ab, input_ab) 
    losses.update(loss.item(), input_gray.size(0))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    batch_time.update(time.time() - end)
    end = time.time()

    # Print MSE
    print('Epoch: [{0}][{1}/{2}]\t'
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            epoch, i+1, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses)) 

  print('Finished training epoch {}'.format(epoch))


# Pre-Processing
# Separating images into test and train datasets
os.makedirs('images/train/class/', exist_ok=True) # 90% of images
os.makedirs('images/val/class/', exist_ok=True)   # 10% of images
for i, file in enumerate(os.listdir('dataset')):
  if i < 5: #Choose 5 images to validate on
    os.rename('dataset/' + file, 'images/val/class/' + file)
  else: # others will be training
    os.rename('dataset/' + file, 'images/train/class/' + file)


use_gpu = torch.cuda.is_available()

model = ColorizationNet()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

#Training
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])
train_imagefolder = GrayscaleImageFolder('images/train', train_transforms)
train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=64, shuffle=True)

# Validation/Testing
val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
val_imagefolder = GrayscaleImageFolder('images/val' , val_transforms)
val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=64, shuffle=False)

if use_gpu: 
  criterion = criterion.cuda()
  model = model.cuda()

os.makedirs('outputs/color', exist_ok=True)
os.makedirs('outputs/gray', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
save_images = True
best_losses = 1e10
epochs = 100


# Train model
for epoch in range(epochs):
  train(train_loader, model, criterion, optimizer, epoch)
  with torch.no_grad():
    losses = validate(val_loader, model, criterion, save_images, epoch)
  if losses < best_losses:
    best_losses = losses
    torch.save(model.state_dict(), 'checkpoints/model-epoch-{}-losses-{:.3f}.pth'.format(epoch+1,losses))

model.eval()

batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

end = time.time()
for i, (input_gray, input_ab, target) in enumerate(val_loader):
    data_time.update(time.time() - end)

    if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

    output_ab = model(input_gray)
    loss = criterion(output_ab, input_ab)
    losses.update(loss.item(), input_gray.size(0))

    for j in range(len(output_ab)):
        save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
        save_name = 'img-{}.jpg'.format(i * val_loader.batch_size + j)
        to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)

    batch_time.update(time.time() - end)
    end = time.time()

    
    # Print MSE
    print('Test: [{0}/{1}]\t'
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            i+1, len(val_loader), batch_time=batch_time, loss=losses))

print('Finished Testing.')
