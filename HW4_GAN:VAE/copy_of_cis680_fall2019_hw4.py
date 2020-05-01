# -*- coding: utf-8 -*-
"""Copy of CIS680_Fall2019_HW4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Uud8FjKw8dK78UeuwsVZdgIklizd_Y7p

# Google Drive

This first code block attaches your google drive and makes a folder structure. You only need to run this when a new VM is assigned to you. To get your code as a single python file go through the following menus File->'Download .py'.

This also downloads the 2 files that contain the dataset and the checkpoint:


https://drive.google.com/open?id=1ABUtpgdWMnMG6S6wLgqxXnAQ-8Fyq5F0

https://drive.google.com/open?id=1ilx871Zws-rS1Ek_ZAC-imD50A8bTQ80
"""

from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import os
from google.colab import drive

# Mount google drive
DRIVE_MOUNT='/content/gdrive'
drive.mount(DRIVE_MOUNT)


# create folder to write data to
CIS680_FOLDER=os.path.join(DRIVE_MOUNT, 'My Drive', 'CIS680_2019')
HOMEWORK_FOLDER=os.path.join(CIS680_FOLDER, 'HW4')
os.makedirs(HOMEWORK_FOLDER, exist_ok=True)

# bootstrap environment into place
from google.colab import auth
auth.authenticate_user()

from googleapiclient.discovery import build
drive_service = build('drive', 'v3')

import io
import os
from googleapiclient.http import MediaIoBaseDownload

def download_file(fn, file_id):
    request = drive_service.files().get_media(fileId=file_id)
    downloaded = io.BytesIO()
    downloader = MediaIoBaseDownload(downloaded, request)
    done = False
    while done is False:
        # _ is a placeholder for a progress object that we ignore.
        # (Our file is small, so we skip reporting progress.)
        _, done = downloader.next_chunk()
    
    downloaded.seek(0)

    folder = fn.split('/')
    if len(folder) > 1:
        os.makedirs(folder[0], exist_ok=True)

    with open(fn, 'wb') as f:
        f.write(downloaded.read())

id_to_fn = {
'1ABUtpgdWMnMG6S6wLgqxXnAQ-8Fyq5F0': 'test.npz',
'1ilx871Zws-rS1Ek_ZAC-imD50A8bTQ80': 'train.npz',
}

# download all files into the vm
for fid, fn in id_to_fn.items():
    print("Downloading %s from %s" % (fn, fid))
    download_file(fn, fid)
    
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

"""# PyTorch Dataset

You know what to do :)
"""

# Part 1 - MNIST

# Separate MNIST into training and testing here
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()
    #,torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
trainset = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 100)
testset = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 100)

# Part 2 - STL-10

# Part 3 - Supplied Data



"""# Model Definition

Define the four models (maybe a good idea to split these out into their own blocks?)
"""



class VAE_Encoder(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(784, 400)
    self.fc2_mean = nn.Linear(400, 20)
    self.fc2_sd = nn.Linear(400,20)

  def forward(self,x):
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.fc1(x))
    mean = F.relu(self.fc2_mean(x))
    sd = F.relu(self.fc2_sd(x))
    return mean, sd

  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

class VAE_Decoder(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc3 = nn.Linear(20,400)
    self.fc4 = nn.Linear(400,784)

  def sample_z(self, mu, sigma, batch_size):
    sampler = torch.distributions.normal.Normal(0,1)
    epsilon = Variable(sampler.sample([batch_size, 20]))
    return mu + torch.exp(sigma / 2) * epsilon

  def forward(self, mu, sigma, batch_size):
    z = self.sample_z(mu,sigma,batch_size) # Need to sample z here from [mu, sigma]
    x = F.relu(self.fc3(z))
    x = F.sigmoid(self.fc4(x))
    return x

class VAE(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = VAE_Encoder()
    self.decoder = VAE_Decoder()
    self.mu = None
    self.sigma = None
  
  def forward(self, x, batch_size = 100):
    self.mu, self.sigma = self.encoder(x)
    x_prime = self.decoder(self.mu, self.sigma, batch_size)
    return x_prime

  def VAE_loss(self, true, pred):
    l = nn.MSELoss(size_average=False)
    err_recon = l(pred, true)
    kl_div = 0.5 * torch.sum(torch.exp(self.sigma) + (self.mu ** 2) - 1. - self.sigma,1).sum()
    return err_recon + kl_div



"""# Train your networks

It might be good to save checkpoints and reload from the most recent. This is due to time constraints inside of colab. (Probably a good idea to train each part in separate blocks?)

## VAE TRAINING
"""

# Train VAE
vae = VAE()
optimizer = optim.Adam(vae.parameters(), lr=.001)
# Let's keep track of 36 images over training
generated_images = torch.zeros(36,5,784)
loss_train = []
loss_test = []
num_epochs = 5
for epoch in range(num_epochs):
  print("Epoch %d/%d" % (epoch+1, num_epochs))
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = vae(inputs)

    # Store results for 36 images over iterations
    if i == 1:
      generated_images[:, epoch, :] = outputs[:36]

    loss = vae.VAE_loss(inputs.view(-1, 784), outputs)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
  loss_train.append(running_loss / float(len(trainloader.dataset)))

  running_loss = 0.0
  for i, data in enumerate(testloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    outputs = vae(inputs)
    loss = vae.VAE_loss(inputs.view(-1, 784), outputs)
    running_loss += loss.item()
  loss_test.append(running_loss / float(len(testloader.dataset)))  
   
  print("Train Loss: ", loss_train[len(loss_train) - 1],
        "Test Loss: ", loss_test[len(loss_test) - 1])
print('Finished Training')

"""## VAE Plotting and Testing"""

plt.plot(loss_train, color = 'blue')
plt.plot(loss_test, color = 'red')
plt.title('Loss Over Training - VAE')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend(labels = ['Training Loss', 'Test Loss'])
plt.show()

# Let's look at the images
for epoch in range(5):
  f, axarr = plt.subplots(6,6)
  for ind in range(36):
    im = generated_images[ind, epoch, :].view(28,28).detach().numpy()
    i = ind // 6
    j = ind % 6
    axarr[i,j].imshow(im)
  axarr[0,0].set_title('VAE Outputs: Epoch ' + str(epoch + 1))

# Now let's look at the original images (train 100-135)
enumerator = enumerate(trainloader)
next(enumerator)
i, [data, labels] = next(enumerator)
actual_images = data[:36, 0, :, :]
f, axarr = plt.subplots(6,6)
for ind in range(36):
  im = actual_images[ind].view(28,28).detach().numpy()
  i = ind // 6
  j = ind % 6
  axarr[i,j].imshow(im)
axarr[0,0].set_title('Actual Images')

root = os.getcwd() + '/gdrive/My Drive/CIS680_2019/HW4'
path = root + '/VAE.pt'
torch.save(vae.state_dict(), path)

"""## DCGAN"""

# Separate STL-10 into training and testing here

transform = transforms.Compose([transforms.Resize(64),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.STL10(root = './data', split = 'train', download = True, transform = transform)
testset = torchvision.datasets.STL10(root = './data', split = 'test', download = True, transform = transform)
train, val = torch.utils.data.random_split(trainset, [4000,1000])

trainloader = torch.utils.data.DataLoader(train, batch_size = 128, shuffle = True)
valloader = torch.utils.data.DataLoader(val, batch_size = 128, shuffle = True)
testloader = torch.utils.data.DataLoader(testset, batch_size = 128, shuffle = False)

enumerator = enumerate(trainloader)
i, [data, labels] = next(enumerator)
plt.imshow(data[6].numpy().transpose((1,2,0)))

plt.imshow(np.transpose(torchvision.utils.make_grid(data.to(device)[:36], nrow = 6, normalize = True, padding=0).cpu(),(1,2,0)))

def reshape_image(image):
    im = np.swapaxes(image, 0 ,1)
    im = np.swapaxes(im, 1, 2)
    return im

class DCGAN_Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv1 = nn.ConvTranspose2d(100, 1024, kernel_size = 4, stride = 1, padding = 0)
        self.bn1 = nn.BatchNorm2d(1024)
        self.tconv2 = nn.ConvTranspose2d(1024, 512, kernel_size = 4, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm2d(512)
        self.tconv3 = nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.tconv4 = nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.tconv5 = nn.ConvTranspose2d(128, 3, kernel_size = 4, stride = 2, padding = 1)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

        self.generate = nn.Sequential(
            self.tconv1,
            self.bn1,
            self.relu,
            self.tconv2,
            self.bn2,
            self.relu,
            self.tconv3,
            self.bn3,
            self.relu,
            self.tconv4,
            self.bn4,
            self.relu,
            self.tconv5,
            self.tanh)
    
    def forward(self, x):
        out = self.generate(x)
        return out

class DCGAN_Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 4, stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1, kernel_size = 4)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace = True)
        

        self.discriminate = nn.Sequential(
            self.conv1,
            self.bn1,
            self.lrelu,
            self.conv2,
            self.bn2,
            self.lrelu,
            self.conv3,
            self.bn3,
            self.lrelu,
            self.conv4,
            self.bn4,
            self.lrelu,
            self.conv5,
            self.sigmoid)

    def forward(self, x):
        out = self.discriminate(x)
        return out

learn_rt = 0.0002
numepochs = 250
generator = DCGAN_Generator().to(device)
discriminator = DCGAN_Discriminator().to(device)
optimizerG = torch.optim.Adam(generator.parameters(), lr = learn_rt, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(discriminator.parameters(), lr = learn_rt, betas=(0.5, 0.999))

#optimizer = torch.optim.Adam(list(generator.parameters()) + list(discriminator.parameters()), lr = learn_rt)

# DEFINE OUR DCGAN TRAIN FUNCTION
def train_DCGAN(generator, discriminator, train_loader, val_loader, optimizerG, optimizerD, num_epochs = 40):  
    # initialize metric arrays
    criterion = nn.BCELoss()
    loss_G_np = np.zeros(num_epochs)
    loss_D_np = np.zeros(num_epochs)
    noise = torch.randn(36, 100, 1, 1).to(device)

    for epoch in range(num_epochs):
        for i, (real, labels) in enumerate(train_loader):
            
            # Update discriminator
            # Train with real batch
            
            # optimizerG.zero_grad()
            optimizerD.zero_grad()
            #discriminator.zero_grad()
            
            real = real.to(device)    
            batch_size = len(real)
            label_real = torch.ones(batch_size)
            output = discriminator(real).reshape([-1])
            loss_D_real = criterion(output.to(device), label_real.to(device))
            loss_D_real.backward()
            D_x = output.mean().item()
                 
            # Train with fake batch
            z = torch.randn(batch_size, 100, 1, 1).to(device)
            label_fake = torch.zeros(batch_size).to(device)
            fake = generator(z)
            fake = fake.to(device)
            output = discriminator(fake).reshape([-1])
            loss_D_fake = criterion(output.to(device), label_fake)
            loss_D_fake.backward()
            D_G_z1 = output.mean().item()

            loss_D = loss_D_real + loss_D_fake             
            optimizerD.step()

            optimizerG.zero_grad()
            #generator.zero_grad()
            # optimizerD.zero_grad()
            # Update generator
            label_real = torch.ones(batch_size).to(device)
            fake = generator(z)
            fake = fake.to(device)
            output = discriminator(fake).reshape([-1])
            D_G_z2 = output.mean().item()
            loss_G = criterion(output, label_real)           
            loss_G.backward()
            optimizerG.step()

            # optimizerG.zero_grad()
            # optimizerD.zero_grad()
            # # Update generator AGAIN
            # z = torch.randn(batch_size, 100, 1, 1).to(device)
            # label_real = torch.ones(batch_size).to(device)
            # fake = generator(z)
            # fake = fake.to(device)
            # output = discriminator(fake)
            # loss_G = criterion(output, label_real)         
            # loss_G.backward()
            # optimizerG.step()
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, num_epochs, i, len(trainloader), D_x, D_G_z1, D_G_z2))
            
        # get loss
        loss_D_np[epoch] = loss_D.item()
        loss_G_np[epoch] = loss_G.item()

        with torch.no_grad():
            out = generator(noise)
        plt.imshow(np.transpose(torchvision.utils.make_grid(out.to(device)[:36], 
                                                            nrow = 6, normalize = True, 
                                                            padding=0).cpu(),(1,2,0)))
        plt.show()
        # for i in range(36):
        #     plt.subplot(6,6,i+1)
        #     plt.imshow(out[i].data.cpu().numpy().transpose((1,2,0)))
        # plt.show()
   
        print('Epoch ' + str(epoch + 1) + ' Loss Gen: ' + str(loss_G_np[epoch]) + ' Loss Dis: ' + str(loss_D_np[epoch]))
    return loss_G_np, loss_D_np

loss_G, loss_D = train_DCGAN(generator, discriminator, trainloader, valloader, optimizerG, optimizerD, num_epochs= numepochs)

# FIGURES FOR THE REPORT
x = list(np.arange(1, 251))
### Figure 1 --> Joint Train Loss: Classifier
fig1 = plt.figure()
plt.plot( x, list(loss_G), marker='', markerfacecolor='blue', 
         markersize=4, color='blue', linewidth=2, label= 'Loss Generator')
plt.plot( x, list(loss_D), marker='', markerfacecolor='blue', 
         markersize=4, color='darkorange', linewidth=2, label='Loss Discriminator')

ax = plt.gca()
ax.tick_params(axis = 'both', labelsize = 16)
plt.xlabel('Epoch', fontsize = 16)
plt.ylim(bottom=0)
plt.legend(fontsize = 12)

#SAVE MODEL PARAMETERS

# PATH ='gdrive/My Drive/CIS680_2019/HW4/gen_2.pt'
# torch.save(generator.state_dict(), PATH)
# PATH ='gdrive/My Drive/CIS680_2019/HW4/dis_2.pt'
# torch.save(discriminator.state_dict(), PATH)

# LOAD MODEL PARAMETERS
PATH ='gdrive/My Drive/CIS680_2019/HW4/gen_2.pt'
generator = DCGAN_Generator()
generator.load_state_dict(torch.load(PATH))
generator.eval()
generator = generator.to(device)

PATH ='gdrive/My Drive/CIS680_2019/HW4/dis_2.pt'
discriminator = DCGAN_Discriminator()
discriminator.load_state_dict(torch.load(PATH))
discriminator.eval()
discriminator = discriminator.to(device)

noise = torch.randn(36, 100, 1, 1).to(device)
out = generator(noise)

#plt.imshow(reshape_image(out[10].reshape(3,64,64).data.cpu().numpy()))

plt.imshow(np.transpose(torchvision.utils.make_grid(out.to(device)[:36], 
                                                            nrow = 6, normalize = True, 
                                                            padding=0).detach().cpu(),(1,2,0)))
plt.show()

"""# Test your networks

Did you remember to cut out a test set? If not you really should, test on images your network has never seen.
"""











import os, pdb
from torchvision import transforms
from google.colab import drive, files
drive.mount('/content/drive')

# Mount google drive
DRIVE_MOUNT='/content/gdrive'
drive.mount(DRIVE_MOUNT)

# create folder to write data to
CIS680_FOLDER=os.path.join(DRIVE_MOUNT, 'My Drive', 'CIS680_2019')
HOMEWORK_FOLDER=os.path.join(CIS680_FOLDER, 'HW3a')
os.makedirs(HOMEWORK_FOLDER, exist_ok=True)

# bootstrap environment into place
from google.colab import auth
auth.authenticate_user()

from googleapiclient.discovery import build
drive_service = build('drive', 'v3')

import io
import os
from googleapiclient.http import MediaIoBaseDownload

def download_file(fn, file_id):
    request = drive_service.files().get_media(fileId=file_id)
    downloaded = io.BytesIO()
    downloader = MediaIoBaseDownload(downloaded, request)
    done = False
    while done is False:
        # _ is a placeholder for a progress object that we ignore.
        # (Our file is small, so we skip reporting progress.)
        _, done = downloader.next_chunk()
    
    downloaded.seek(0)

    folder = fn.split('/')
    if len(folder) > 1:
        os.makedirs(folder[0], exist_ok=True)

    with open(fn, 'wb') as f:
        f.write(downloaded.read())

id_to_fn = {
'1ABUtpgdWMnMG6S6wLgqxXnAQ-8Fyq5F0': 'test.npz',
'1ilx871Zws-rS1Ek_ZAC-imD50A8bTQ80': 'train.npz',
}

# download all files into the vm
for fid, fn in id_to_fn.items():
    print("Downloading %s from %s" % (fn, fid))
    download_file(fn, fid)

"""# PyTorch Dataset

You know what to do :)
"""

from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from torch.nn import Conv2d, BatchNorm2d, ConvTranspose2d
from torch.nn.functional import leaky_relu
from torchsummary import summary
import numpy as np

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

class ToCuda(object):
    """Put Tensor on Cuda"""
    def __call__(self, sample):
        return sample.cuda()

class SRGANDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, transform=None):
        with np.load(images_path, allow_pickle=True, encoding='bytes') as data:
            self.images = data['arr_0'].astype(np.uint8)
        self.transform = transform

    def __len__(self):
        return len(self.images)
   
    def get_image(self, idx):
        return self.images[idx]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.transform:
            return self.transform(self.images[idx])
        else:
            return self.images[idx]

transform = transforms.Compose([
    transforms.ToTensor(), 
    ToCuda(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

"""
Useful transform to go from normalized image tensor [-1,1] to original image space [0,1]
"""
unnorm = transforms.Compose([
    UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = SRGANDataset("train.npz", transform=transform)
testset = SRGANDataset("test.npz", transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# gen = Generator().cuda()
# A = torch.empty((1,3,64,64), device='cuda')
# Z = gen(A)

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 32, 4, stride=2, padding=1) 
        self.conv2 = Conv2d(32, 64, 4, stride=2, padding=1) 
        self.conv3 = Conv2d(64, 128, 4, stride=2, padding=1) 
        self.conv4 = Conv2d(128, 256, 4, stride=2, padding=1) 
        self.conv5 = Conv2d(256, 512, 4, stride=2, padding=1) 

        self.bn1 = BatchNorm2d(32)
        self.bn2 = BatchNorm2d(64)
        self.bn3 = BatchNorm2d(128)
        self.bn4 = BatchNorm2d(256) 
        self.bn5 = BatchNorm2d(512)
    
    def forward(self, X):
        skip_feats = []
        skip_feats.append(X)
        X = leaky_relu(self.bn1(self.conv1(X)))
        skip_feats.append(X)
        X = leaky_relu(self.bn2(self.conv2(X)))
        skip_feats.append(X)
        X = leaky_relu(self.bn3(self.conv3(X)))
        skip_feats.append(X)
        X = leaky_relu(self.bn4(self.conv4(X)))
        skip_feats.append(X)
        X = leaky_relu(self.bn5(self.conv5(X)))
        return X, skip_feats

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvTranspose2d(512, 256, 4, stride=2, padding=1) 
        self.conv2 = ConvTranspose2d(512, 128, 4, stride=2, padding=1) 
        self.conv3 = ConvTranspose2d(256, 64, 4, stride=2, padding=1) 
        self.conv4 = ConvTranspose2d(128, 32, 4, stride=2, padding=1) 
        self.conv5 = ConvTranspose2d(64, 16, 4, stride=2, padding=1) 
        self.conv6 = ConvTranspose2d(19, 3, 1, stride=1, padding=0) 

        self.bn1 = BatchNorm2d(256)
        self.bn2 = BatchNorm2d(128)
        self.bn3 = BatchNorm2d(64)
        self.bn4 = BatchNorm2d(32) 
        self.bn5 = BatchNorm2d(16)        

    def forward(self, X, skip_feats):
        X = leaky_relu(self.bn1(self.conv1(X)))
        X = leaky_relu(self.bn2(self.conv2(torch.cat([X, skip_feats[-1]], dim=1))))
        X = leaky_relu(self.bn3(self.conv3(torch.cat([X, skip_feats[-2]], dim=1))))
        X = leaky_relu(self.bn4(self.conv4(torch.cat([X, skip_feats[-3]], dim=1))))
        X = leaky_relu(self.bn5(self.conv5(torch.cat([X, skip_feats[-4]], dim=1))))
        X = torch.tanh(self.conv6(torch.cat([X, skip_feats[-5]], dim=1)))
        return X


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, X):
        X = self.decoder(*self.encoder(X))
        return X

class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_shape = (1,1,1)

        self.conv1 = Conv2d(3, 32, 4, stride=2, padding=1) 
        self.conv2 = Conv2d(32, 64, 4, stride=2, padding=1) 
        self.conv3 = Conv2d(64, 128, 4, stride=2, padding=1) 
        self.conv4 = Conv2d(128, 256, 4, stride=2, padding=1) 
        self.conv5 = Conv2d(256, 512, 4, stride=2, padding=1) 
        self.conv6 = Conv2d(512, 1, 3, stride=2, padding=1)

        self.bn1 = BatchNorm2d(32)
        self.bn2 = BatchNorm2d(64)
        self.bn3 = BatchNorm2d(128)
        self.bn4 = BatchNorm2d(256) 
        self.bn5 = BatchNorm2d(512)

    def forward(self, X):
        X = leaky_relu(self.bn1(self.conv1(X)))
        X = leaky_relu(self.bn2(self.conv2(X)))
        X = leaky_relu(self.bn3(self.conv3(X)))
        X = leaky_relu(self.bn4(self.conv4(X)))
        X = leaky_relu(self.bn5(self.conv5(X)))
        X = torch.sigmoid(self.conv6(X))
        return X

"""# Train your networks

It might be good to save checkpoints and reload from the most recent. This is due to time constraints inside of colab. (Probably a good idea to train each part in separate blocks?)
"""

generator = Generator().cuda()
discriminator = Discriminator().cuda()

criterion_GAN = nn.BCELoss().cuda()
criterion_pixel = nn.MSELoss().cuda()

optimizer_GEN = torch.optim.Adam(generator.parameters(), lr=2e-4)
optimizer_DIS = torch.optim.Adam(discriminator.parameters(), lr=2e-4)

USE_PRETRAINED = False
if USE_PRETRAINED:
    try:
        generator.load_state_dict(torch.load(os.path.join(HOMEWORK_FOLDER, 'generator_2019-12-04_14:38:34.ckpt')))
        discriminator.load_state_dict(torch.load(os.path.join(HOMEWORK_FOLDER, 'discriminator_2019-12-04_14:38:34.ckpt')))
        print("Loaded weights for all networks")
    except:
        print("One of the checkpoints was not found")

SAVE_EPOCH_CHECKPOINTS = True
LOG_LOSS_MINI_BATCHES = 300

generator.train()
discriminator.train()

loss_train_gen = []
loss_train_dis = []

# Save images for report
sampled_images = []

num_epochs = 40
epochs_to_sample = np.round(np.linspace(0, num_epochs, 6, endpoint=False))

track_image = trainset.__getitem__(5).unsqueeze(0)
lr_track = F.interpolate(track_image, size=(16,16))
lr_full_track = F.interpolate(lr_track, size=(64,64), mode="bilinear", align_corners=False)

for epoch in range(num_epochs):
    print("Epoch [%d/%d]" % (epoch+1, num_epochs))

    SAMPLE_IMAGE = epoch in epochs_to_sample
    if SAMPLE_IMAGE:
        print("Saving image in epoch [%d/%d]" % (epoch+1, num_epochs))

    running_gen, running_dis = 0.0, 0.0
    
    for i, images in enumerate(trainloader):
                
        real = torch.ones((images.size(0), *discriminator.output_shape), requires_grad=False).cuda()
        fake = torch.zeros((images.size(0), *discriminator.output_shape), requires_grad=False).cuda()        

        lr = F.interpolate(images, size=(16,16))
        lr_full = F.interpolate(lr, size=(64,64), mode="bilinear", align_corners=False)

        # (1) GENERATOR step
        optimizer_GEN.zero_grad()
                
        # Generate HR fake images
        hr_fake = generator(lr_full)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(hr_fake), real)

        # Content loss (MSE on pixel content)
        loss_pixel = criterion_pixel(hr_fake, images.detach())

        # Total loss
        loss_GEN = loss_pixel + (2e-4 * loss_GAN)

        loss_GEN.backward()
        optimizer_GEN.step()

        # Show the upsample of the current generator on the fixed image
        if i == 1 and SAMPLE_IMAGE:
            sampled_images.append(
                unnorm(generator(lr_full_track)).squeeze().detach().cpu())

        # (2) DISCRIMINATOR step
        
        optimizer_DIS.zero_grad()

        loss_real = criterion_GAN(discriminator(images), real)
        loss_fake = criterion_GAN(discriminator(hr_fake.detach()), fake)

        # Adverserial loss of real + generated images
        loss_DIS = loss_real + loss_fake

        loss_DIS.backward()
        optimizer_DIS.step()

        # Accumulate generator/discriminator losses
        running_gen += loss_GEN.item()
        running_dis += loss_DIS.item()

        # Log losses locally
        if i % LOG_LOSS_MINI_BATCHES == (LOG_LOSS_MINI_BATCHES - 1):   
            print("Epoch: {}, Batch: {} | Gen: {:.3f}| Dis {:.3f}".format(
                epoch + 1, i + 1,
                running_gen / LOG_LOSS_MINI_BATCHES,
                running_dis / LOG_LOSS_MINI_BATCHES,
            ))
            loss_train_gen.append(running_gen / LOG_LOSS_MINI_BATCHES)
            loss_train_dis.append(running_dis / LOG_LOSS_MINI_BATCHES)
            running_gen, running_dis = 0.0, 0.0
    
    if SAVE_EPOCH_CHECKPOINTS and (epoch % 10 == 9):
        print("Saving checkpoint in epoch [%d/%d]" % (epoch+1, num_epochs))
        # Save the generator
        checkpoint_file = datetime.now().strftime("generator_%Y-%m-%d_%H:%M:%S.ckpt")
        torch.save(generator.state_dict(), os.path.join(HOMEWORK_FOLDER, checkpoint_file))

        checkpoint_file = datetime.now().strftime("discriminator_%Y-%m-%d_%H:%M:%S.ckpt")
        torch.save(discriminator.state_dict(), os.path.join(HOMEWORK_FOLDER, checkpoint_file))

"""### Show Tracked Image over Training"""

grid_img = make_grid(unnorm(torch.stack(sampled_images)))
plt.imshow(grid_img.permute(1, 2, 0))

# plt.savefig("trackimage_0_40.png", dpi=600)
# files.download("trackimage_0_40.png")

"""# Test your networks

Did you remember to cut out a test set? If not you really should, test on images your network has never seen.

#### LPIPS Metrics
"""

!git clone https://github.com/richzhang/PerceptualSimilarity
!cd PerceptualSimilarity
import models
criterion_lpips = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])

# Instantiate the LPIPS metric
criterion_pixel = nn.MSELoss().cuda()

generator.eval()
discriminator.eval()

mse = np.empty(len(testloader))
lpips = np.empty(len(testloader))

for i, images in enumerate(testloader):
            
    lr = F.interpolate(images, size=(16,16))
    lr_full = F.interpolate(lr, size=(64,64), mode="bilinear", align_corners=False)

    # Generate HR fake images
    hr_fake = generator(lr_full)

    # MSE on pixel content
    loss_pixel = criterion_pixel(hr_fake, images.detach())
    # LPIPS
    loss_lpips = criterion_lpips.forward(hr_fake, images.detach())

    mse[i] = loss_pixel.item()
    lpips[i] = loss_lpips.item()

print("Average MSE: {:.3f}".format(np.mean(mse)))
print("Average LPIPS: {:.3f}".format(np.mean(lpips)))

def show(img):
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

track_image = trainset.__getitem__(5).unsqueeze(0)
lr_track = F.interpolate(track_image, size=(16,16))
lr_full_track = F.interpolate(lr_track, size=(64,64), mode="bilinear", align_corners=False)

grid_img = make_grid([
    unnorm(track_image).squeeze().detach().cpu(),
    unnorm(lr_full_track).squeeze().detach().cpu()         
])

plt.imshow(grid_img.permute(1, 2, 0))

plt.savefig("ground_truth.png", dpi=600)
files.download("ground_truth.png")

"""# Visualize GAN Output"""

imgs = []
for i in range(20, 30):
    item = trainset.__getitem__(i).unsqueeze(0)

    down = F.interpolate(item, size=(16,16))
    up = F.interpolate(down, size=(64,64), mode="bilinear", align_corners=False)
    hr_fake = generator(up.cuda())

    imgs.append(unnorm(up).squeeze().detach().cpu())
    imgs.append(unnorm(hr_fake).squeeze().detach().cpu())
    imgs.append(unnorm(item).squeeze().detach().cpu())

grid_img = make_grid(imgs, nrow=3)
plt.imshow(grid_img.permute(1, 2, 0))

plt.savefig("results_0_40.png", dpi=600)
files.download("results_0_40.png")

# loss_DIS = (loss_real + loss_fake) / 2