# -*- coding: utf-8 -*-
"""CIS680_HW1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vNSLwXu6O1L9yH3oEU0bHTFB_TFopW9N
"""

# %matplotlib inline
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from google.colab import files
from google.colab import drive
drive.mount('/content/gdrive')

import torchvision
import torchvision.transforms as transforms
from matplotlib import cm

from torch.nn import Conv2d, AvgPool2d, BatchNorm2d, BatchNorm1d
import torch.nn.functional as F
from torchsummary import summary

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import os
print(os.getcwd())

"""Part 1.1 - plot of sigmoid function"""

# x = 1.0
# w = np.arange(-2.0, 2.0, 0.1)
# b = np.arange(-2.0, 2.0, 0.1)
# W_n, B_n = np.meshgrid(w, b)
# sig = 1.0 / (1.0 + np.exp(-(W_n*x + B_n)))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(W_n, B_n, sig)

w = torch.arange(-5.0, 5.1, 0.1)
b = torch.arange(-5.0, 5.1, 0.1)

W, B = torch.meshgrid([w, b])
W, B = W.t(), B.t()
W.requires_grad = True
B.requires_grad = True

x = torch.ones(B.shape)

Z = W*x + B
sig = torch.sigmoid(Z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(
    W.detach().numpy(),
    B.detach().numpy(),
    sig.detach().numpy(), cmap=cm.coolwarm)

ax.set_xlabel("Weight")
ax.set_ylabel("Bias")
ax.set_zlabel("Output")
plt.title("Sigmoid Activation")

plt.savefig("Sigmoid.png", dpi=600)
files.download("Sigmoid.png")

"""Part 1.2 - L2 Loss"""

y = torch.tensor(0.5).repeat(W.shape)
l2_loss = torch.pow((y - sig), 2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(
    W.detach().numpy(),
    B.detach().numpy(),
    l2_loss.detach().numpy(),
    cmap=cm.coolwarm)

ax.set_xlabel("Weight")
ax.set_ylabel("Bias")
ax.set_zlabel("L2 Loss")
plt.title("L2 Loss")

plt.savefig("L2.png", dpi=600)
files.download("L2.png")

"""Part 1.3 - L2 Loss gradient"""

avg_loss = l2_loss.sum()
avg_loss.backward()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W.detach().numpy(),
                B.detach().numpy(),
                W.grad.numpy(),
                cmap=cm.coolwarm)
ax.set_xlabel("Weight")
ax.set_ylabel("Bias")
ax.set_zlabel("Gradient (d/dW)")
plt.title("L2 Gradient: d/dW")

plt.savefig("L2Grad.png", dpi=600)
files.download("L2Grad.png")

# s = sig.clone().detach()
# l2 = l2_loss.clone().detach()
# man_comp = l2 * (s * (1-s))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(W.detach().numpy(),
#                 B.detach().numpy(),
#                 man_comp.numpy(),
#                 cmap=cm.coolwarm)
# ax.set_xlabel("Weight")
# ax.set_ylabel("Bias")
# ax.set_zlabel("Gradient")

# Pretty much what you get with auto-grad computation if you take torch.abs of the gradients

w = torch.arange(-5.0, 5.1, 0.1)
b = torch.arange(-5.0, 5.1, 0.1)

W, B = torch.meshgrid([w, b])
W, B = W.t(), B.t()
W.requires_grad = True
B.requires_grad = True

x = torch.ones(B.shape)

Z = W*x + B
sig = torch.sigmoid(Z)

y = torch.tensor(0.5).repeat(W.shape)
ce_loss = -1 * ((y * torch.log(sig)) + (1-y) * (torch.log(1-sig)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(
    W.detach().numpy(),
    B.detach().numpy(),
    ce_loss.detach().numpy(),
    cmap=cm.coolwarm)

ax.set_xlabel("Weight")
ax.set_ylabel("Bias")
ax.set_zlabel("Cross-Entropy Loss")
plt.title("Cross-Entropy Loss")

plt.savefig("CELoss.png", dpi=600)
files.download("CELoss.png")

avg_loss = ce_loss.sum()
avg_loss.backward()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W.detach().numpy(),
                B.detach().numpy(),
                W.grad.numpy(),
                cmap=cm.coolwarm)
ax.set_xlabel("Weight")
ax.set_ylabel("Bias")
ax.set_zlabel("Gradient (d/dW)")
plt.title("Cross-Entropy Gradient: d/dW")

plt.savefig("CEGrad.png", dpi=600)
files.download("CEGrad.png")

"""Part 1.6

1) L2 loss is the squared difference between the predicted output and the true output. Cross entropy loss measures the scaled log innacuracy of the predicted output by the true output. 

2) The gradient of L2 goes to 0 for wildly inaccurate predictions (regions with high loss). Likewise, the magnitude of the gradient of cross-entropy increases as the loss increases. Both gradients are predictably zero when the loss is zero.

3) L2 loss will be highly inefficient for classification tasks, as the gradient for incorrect predictions may be close to zero. Likewise, cross-entropy loss has the desired behavior for classification tasks - large gradients for incorrect predictions, near zero gradient for correct predictions. Cross-entropy is clearly ill-conditioned for regression tasks where the target is often any real number, not a classification label.

Part 2 - Solving XOR with a 2-layer Perceptron
"""

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], requires_grad=True)
y = torch.tensor([[0], [1], [1], [0]], requires_grad=False).type(torch.FloatTensor)

class Net(torch.nn.Module):
  def __init__(self):
      super(Net, self).__init__()
      self.hidden = torch.nn.Linear(X.shape[1], 2)
      self.out = torch.nn.Linear(2, 1)

  def forward(self, x):
      x = torch.tanh(self.hidden(x))
      x = torch.sigmoid(self.out(x))
      return x

net = Net()

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

running_loss = 0.0
accuracies = np.zeros(100)
torch.manual_seed(7)

for epoch in range(250):  # loop over the dataset multiple times
           
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    if epoch % 75 == 0:
      accuracies[epoch//75] == (outputs.round() == y).sum().item() / y.shape[0]
      
      h = net.hidden.weight.data.numpy()
      b = net.hidden.bias.data.numpy()
      
      # The gist is get the decision boundary from the weights vector
      # Weights vector defined as line between two weights points
      # Decision boundary orthogonal to weight vector
     
      mid_x = (h[1,0] + h[0,0])/2
      mid_y = ((h[1,1] + h[0,1]))/2
      
      p1 = (mid_x, mid_y)
      p2 = (mid_x - h[1,1] + h[0,1], mid_y + h[1,0] - h[0, 0])
      
      m = (p2[1] - p1[1]) / (p2[0] - p1[0])
      c = (p2[1] - (m * p2[0]))
      
      h_x_axis = np.arange(0, 1., .1)
      
      x_axis = np.arange(-1, 1, 0.1)
      y_axis = m * x_axis + c

      one_ind = [1, 2]
      zero_ind = [0, 3]
      mapped = np.tanh(X.detach().numpy() @ h.T + b)
      
      fig = plt.figure()
      plt.scatter(mapped[one_ind, 0], mapped[one_ind, 1], marker='x', c='r', s=120)
      plt.scatter(mapped[zero_ind, 0], mapped[zero_ind, 1], marker='o', c='b', s=120)
      plt.plot(x_axis, y_axis)
        
      plt.title("Decision boundary: {} iters".format(epoch))
      plt.xlabel("h(x1)")
      plt.ylabel("h(x2)")
  
      # UNCOMMENT TO SAVE IMAGES TO FILE
#       plt.savefig("decision_{}.png".format(epoch), dpi=600)
#       files.download("decision_{}.png".format(epoch))

"""Part 3 - Train a CNN"""

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

dataiter = iter(trainloader)
images, labels = dataiter.next()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg[0], (0, 1)), cmap=cm.viridis)
    plt.show()
    return

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % labels[j].item() for j in range(10)))

class ConvNet(torch.nn.Module):
  def __init__(self):
      super(ConvNet, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=5//2)
      self.conv1_bn = BatchNorm2d(32)
      self.pool1 = nn.AvgPool2d(2, stride=2, padding=0)
      self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=5//2)
      self.conv2_bn = BatchNorm2d(32)
      self.pool2 = nn.AvgPool2d(2, stride=2, padding=0)
      self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=5//2)
      self.conv3_bn = BatchNorm2d(64)
      self.pool3 = nn.AvgPool2d(2, stride=2, padding=0)
      self.fcn1 = nn.Linear(576, 64)
      self.fcn1_bn = BatchNorm1d(64)
      self.fcn2 = nn.Linear(64, 10)

  def forward(self, x):
      x = F.relu(self.conv1_bn(self.conv1(x)))
      x = self.pool1(x)
      x = F.relu(self.conv2_bn(self.conv2(x)))
      x = self.pool2(x)
      x = F.relu(self.conv3_bn(self.conv3(x)))
      x = self.pool3(x)
      x = x.view(x.size(0), -1)
      x = F.relu(self.fcn1_bn(self.fcn1(x)))
      x = self.fcn2(x)
      return F.softmax(x, dim=1)

PATH = os.getcwd() + '/gdrive/My Drive/Colab Notebooks/CIS680/HW1/conv_net.pt'
LOADED_SAVED_MODEL = False

net = ConvNet()

try:
  net.load_state_dict(torch.load(PATH))
  print("Saved model weights loaded")
  LOADED_SAVED_MODEL = True
except:
  print("Saved model not found")

if torch.cuda.is_available():
  net.cuda()

summary(net, (1, 28, 28))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

if not LOADED_SAVED_MODEL:
  
  train_accs = []
  test_accs = []
  
  for epoch in range(8):  # loop over the dataset multiple times

      running_loss = 0.0
      for i, data in enumerate(trainloader, 0):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data

          inputs = inputs.cuda()
          labels = labels.cuda()

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
          if i % 250 == 249:    # print every 249 mini-batches
            
              print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 250))
              running_loss = 0.0
              
      train_accs.append(train_accuracy())
      test_accs.append(test_accuracy())
  
  print('Finished Training')  
else:
  print("Using pre-trained model")

plt.title("Train/Test Accuracy")
plt.plot(range(len(train_accs)), train_accs)
plt.plot(range(len(test_accs)), test_accs)
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.savefig("Train_test.png".format(epoch), dpi=600)
files.download("Train_test.png".format(epoch))

def train_accuracy():
  return accuracy(trainloader)

def test_accuracy():
  return accuracy(testloader)

def accuracy(loader):
  correct = 0
  total = 0
  with torch.no_grad():
      for data in loader:
          images, labels = data
          images = images.cuda()
          labels = labels.cuda()
          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
 
  return correct / total

"""This was my initial (non-experimental) solution to Part 4. I made a second attempt at Part 4 in another file that is better, but overly complicated for this task. 

See the other file for another writeup of Part 4.

Part 4.1 - Pick and image that we classified correctly
"""

def is_equal (ind1, ind2):
  return ind1 == ind2

def conf_greater_90(confidence):
  return confidence > 0.90

def stop_criteria(ind1, ind2, confidence):
  """Criteria to stop adverserial training
     Returns: Boolean
        True -> Don't stop
        False -> stop
  """
  if is_equal(ind1, ind2):
    return False
  
  return conf_greater_90(confidence)

def generate_adverserial(clean_img, clean_label, epsilon=0.03):

  img_ep = torch.zeros(clean_img.shape, requires_grad=True)
  img_pet = (clean_img + img_ep).clone()

  target = clean_label.clone().cuda()
  confidences = torch.zeros(adv_pred.shape).cuda()

  net.eval()

  epoch = 0
  while (epoch < 100):

      outputs = net(img_pet.cuda())

      confidence, pred = outputs.max(1)

      loss = criterion(outputs, clean_label.cuda())
      loss.backward()

      ep_grad = img_ep.grad.data.sign()
      # non-decreasing constraint
      ep_grad[ep_grad < 0.] = 0. 

      img_ep.data += epsilon * ep_grad
      img_pet.data = clean_img.data + img_ep.data

      # clamp value constraint
      img_pet.data = torch.clamp(img_pet.data, -1., 1.) 

      epoch += 1

      if stop_criteria(pred, target, confidence):
        break
  
      # Uncomment for step-by-step loss/label tracking. It's cool.
    
#       print('[{:d}] loss: {:.3f} | labels: {} | conf: {:.3f}'.format(
#           epoch,
#           loss,
#           pred.item(),
#           confidence.item()))

  print('Finished Training')
  print('--------------------------------')
  print("Original label: {}".format(target.item()))
  print("Final label: {}".format(pred.item()))
  print("Final confidence: {}".format(confidence.item()))
  
  return img_pet, pred, confidence

dataiter = iter(testloader)
images, labels = dataiter.next()
n_adver = 3
epsilon = 0.03

outputs = net(images.cuda())
_, predictions = torch.max(outputs.data, 1)

# Pick out the first two thing we labeled correctly
correct_pred_idx = (predictions == labels.cuda()).nonzero()[:n_adver].reshape(-1)

# Pick out the images and labels
clean_img = images[correct_pred_idx]
clean_label = labels[correct_pred_idx]

imshow(torchvision.utils.make_grid(clean_img.clone().cpu()))


print("True labels: {}".format(clean_label.reshape(-1).clone().cpu().numpy()))

# Display several examples of adveserial generation
for idx in range(n_adver):
  img_pet, pred, confidence = generate_adverserial(
      clean_img[idx].unsqueeze(0), 
      clean_label[idx].unsqueeze(0),
      epsilon=epsilon)
  
  original = clean_img[idx].unsqueeze(0).clone()
  perturbed = img_pet.detach().cpu().clone()
  
  grid_out = torch.cat((original, perturbed), 0)

  imshow(torchvision.utils.make_grid(grid_out))

"""**Part 4.2 - Generating Adversial Images with a Specified Target**

Loss function doesn't change, we just change the target labels, the stopping criteria, and multiply the loss
"""

def is_unequal (ind1, ind2):
  return ind1 != ind2

def conf_greater_90(confidence):
  return confidence > 0.90

def stop_criteria(ind1, ind2, confidence):
  """Criteria to stop adverserial training
     Returns: Boolean
        True -> Don't stop
        False -> stop
  """
  if is_unequal(ind1, ind2):
    return False
  
  return conf_greater_90(confidence)

def generate_target_adverserial(clean_img, original_label, target_label, epsilon=0.03):

  img_ep = torch.zeros(clean_img.shape, requires_grad=True)
  img_pet = (clean_img + img_ep).clone()

  target = torch.tensor([target_label]).cuda()
  confidences = torch.zeros(adv_pred.shape).cuda()

  net.eval()

  epoch = 0
  while (epoch < 100):

      outputs = net(img_pet.cuda())

      confidence, pred = outputs.max(1)

      # Force the gradient towards the label we want
      # target in loss is the target label, not the true label
      loss = -1 * criterion(outputs, target)
      
      loss.backward()

      ep_grad = img_ep.grad.data.sign()
      
      # non-decreasing constraint
      ep_grad[ep_grad < 0.] = 0. 

      img_ep.data += epsilon * ep_grad
      img_pet.data = clean_img.data + img_ep.data

      # clamp value constraint
      img_pet.data = torch.clamp(img_pet.data, -1., 1.) 

      epoch += 1

      if stop_criteria(pred, target, confidence):
        break
  
      # Uncomment for step-by-step loss/label tracking. It's cool.
    
#       print('[{:d}] loss: {:.3f} | labels: {} | conf: {:.3f}'.format(
#           epoch,
#           loss,
#           pred.item(),
#           confidence.item()))

  print('Finished Training')
  print('--------------------------------')
  print("Original label: {}".format(original_label.item()))
  print("Final label: {}".format(pred.item()))
  print("Final confidence: {}".format(confidence.item()))
  
  return img_pet, pred, confidence

dataiter = iter(testloader)
images, labels = dataiter.next()
n_adver = 3
target_label = 6
epsilon = 0.02

outputs = net(images.cuda())
_, predictions = torch.max(outputs.data, 1)

# Pick out the first two thing we labeled correctly
correct_pred_idx = (predictions == labels.cuda()).nonzero()[:n_adver].reshape(-1)

# Pick out the images and labels
clean_img = images[correct_pred_idx]
clean_label = labels[correct_pred_idx]

imshow(torchvision.utils.make_grid(clean_img.clone().cpu()))


print("True labels: {}".format(clean_label.reshape(-1).clone().cpu().numpy()))

# Display several examples of adveserial generation
for idx in range(n_adver):
  img_pet, pred, confidence = generate_target_adverserial(
      clean_img[idx].unsqueeze(0), 
      clean_label[idx].unsqueeze(0),
      target_label,
      epsilon=epsilon)
  
  original = clean_img[idx].unsqueeze(0).clone()
  perturbed = img_pet.detach().cpu().clone()
  
  grid_out = torch.cat((original, perturbed), 0)

  imshow(torchvision.utils.make_grid(grid_out))

"""**4.3) Re-training the model. See the OTHER FILE. For this one, I generated batches of adverserial images using an experimental method. Writeup references code in the other file.**"""