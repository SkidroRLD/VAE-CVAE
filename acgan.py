"""Assignment 9
Part 2: AC-GAN

NOTE: Feel free to check: https://arxiv.org/pdf/1610.09585.pdf

NOTE: Write Down Your Info below:

    Name:

    CCID:

    Auxiliary Test Accuracy on Cifar10 Test Set:


"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt

def compute_score(acc, min_thres, max_thres):
    if acc <= min_thres:
        base_score = 0.0
    elif acc >= max_thres:
        base_score = 100.0
    else:
        base_score = float(acc - min_thres) / (max_thres - min_thres) \
                     * 100
    return base_score


# -----
# AC-GAN Build Blocks

# #####
# TODO: Complete the generator architecture
# #####

class Generator(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3, num_classes=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.out_channels = out_channels

        # #####
        # TODO: Complete the generator architecture
        # #####


        
    def forward(self, z, y):
        # #####
        # TODO: Complete the generator architecture
        # #####
        raise NotImplementedError

# #####
# TODO: Complete the Discriminator architecture
# #####

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        # #####
        # TODO: Complete the discriminator architecture
        # #####



    def forward(self, x):
        # #####
        # TODO: Complete the discriminator architecture
        # #####
        raise NotImplementedError
        

# -----
# Hyperparameters
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# NOTE: Feel free to change the hyperparameters as long as you meet the marking requirement
batch_size = 256
workers = 6
latent_dim = 128
lr = 0.001
num_epochs = 150
validate_every = 1
print_every = 100

save_path = os.path.join(os.path.curdir, "visualize", "gan")
if not os.path.exists(os.path.join(os.path.curdir, "visualize", "gan")):
    os.makedirs(os.path.join(os.path.curdir, "visualize", "gan"))
ckpt_path = 'acgan.pt'

# -----
# Dataset
# NOTE: Data is only normalized to [0, 1]. THIS IS IMPORTANT!!!
tfms = transforms.Compose([
    transforms.ToTensor(), 
    ])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True,
    transform=tfms)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True, 
    num_workers=workers)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True,
    transform=tfms)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=batch_size,
    shuffle=False, 
    num_workers=workers)


# -----
# Model
# #####
# TODO: Initialize your models HERE.
# #####
generator = None

discriminator = None

# -----
# Losses

# #####
# TODO: Initialize your loss criterion.
# #####

adv_loss = None
aux_loss = None

if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    adv_loss = adv_loss.cuda()
    aux_loss = aux_loss.cuda()

# Optimizers for Discriminator and Generator, separate

# #####
# TODO: Initialize your optimizer(s).
# #####

optimizer_D = None
optimizer_G = None


# -----
# Train loop

def denormalize(x):
    """Denomalize a normalized image back to uint8.
    """
    # #####
    # TODO: Complete denormalization.
    # #####
    raise NotImplementedError

# For visualization part
# Generate 20 random sample for visualization
# Keep this outside the loop so we will generate near identical images with the same latent featuresper train epoch
random_z = None
random_y = None


# #####
# TODO: Complete train_step for AC-GAN
# #####

def train_step(x, y):
    """One train step for AC-GAN.
    You should return loss_g, loss_d, acc_d, a.k.a:
        - average train loss over batch for generator
        - average train loss over batch for discriminator
        - auxiliary train accuracy over batch
    """
    raise NotImplementedError

def test(
    test_loader,
    ):
    """Calculate accuracy over Cifar10 test set.
    """
    size = len(test_loader.dataset)
    corrects = 0

    discriminator.eval()
    with torch.no_grad():
        for inputs, gts in test_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                gts = gts.cuda()

            # Forward only
            _, outputs = discriminator(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == gts.data)

    acc = corrects / size
    print("Test Acc: {:.4f}".format(acc))
    return acc


g_losses = []
d_losses = []
best_acc_test = 0.0

for epoch in range(1, num_epochs + 1):
    generator.train()
    discriminator.train()

    avg_loss_g, avg_loss_d = 0.0, 0.0
    for i, (x, y) in enumerate(train_loader):
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        # train step
        loss_g, loss_d, acc_d = train_step(x, y)
        avg_loss_g += loss_g * x.shape[0]
        avg_loss_d += loss_d * x.shape[0]

        # Print
        if i % print_every == 0:
            print("Epoch {}, Iter {}: LossD: {:.6f} LossG: {:.6f}, D_acc {:.6f}".format(epoch, i, loss_g, loss_d, acc_d))

    g_losses.append(avg_loss_g / len(train_dataset))
    d_losses.append(avg_loss_d / len(train_dataset))

    # Save
    if epoch % validate_every == 0:
        acc_test = test(test_loader)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            # Wrap things to a single dict to train multiple model weights
            state_dict = {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                }
            torch.save(state_dict, ckpt_path)
            print("Best model saved w/ Test Acc of {:.6f}.".format(best_acc_test))


        # Do some reconstruction
        generator.eval()
        with torch.no_grad():
            # Forward
            xg = generator(random_z, random_y)
            xg = denormalize(xg)

            # Plot 20 randomly generated images
            plt.figure(figsize=(10, 5))
            for p in range(20):
                plt.subplot(4, 5, p+1)
                plt.imshow(xg[p])
                plt.text(0, 0, "{}".format(classes[random_y[p].item()]), color='black',
                            backgroundcolor='white', fontsize=8)
                plt.axis('off')

            plt.savefig(os.path.join(os.path.join(save_path, "E{:d}.png".format(epoch))), dpi=300)
            plt.clf()
            plt.close('all')

        # Plot losses
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(g_losses, label="G")
        plt.plot(d_losses, label="D")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xlim([1, epoch])
        plt.legend()
        plt.savefig(os.path.join(os.path.join(save_path, "loss.png")), dpi=300)

# Just for you to check your Part 2 score
score = compute_score(best_acc_test, 0.65, 0.69)
print("Your final accuracy:", best_acc_test)
print("Your Assignment Score:", score)