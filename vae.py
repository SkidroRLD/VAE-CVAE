"""Assignment 9
Part 1: Variational Autoencoder + Conditional Variational Autoencoder

NOTE: Feel free to check: https://arxiv.org/pdf/1512.09300.pdf

NOTE: Write Down Your Info below:

    Name: Shivam Rathore

    CCID: srathore

    Average Reconstruction Loss per Sample over Cifar10 Test Set: VAE - 0.74802392578125 CVAE -  


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
import math

from ssim import SSIM

def main():
    # torch.backends.cudnn.enabled = False
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")

    def compute_score(loss, min_thres, max_thres):
        if loss <= min_thres:
            base_score = 100.0
        elif loss >= max_thres:
            base_score = 0.0
        else:
            base_score = (1 - float(loss - min_thres) / (max_thres - min_thres)) \
                        * 100
        return base_score

    # -----
    # VAE Build Blocks

    # #####
    # TODO: Complete the encoder architecture
    # #####

    class Encoder(nn.Module):
        def __init__(
            self,
            latent_dim: int = 128,
            in_channels: int = 3,
            ):
            super(Encoder, self).__init__()
            self.latent_dim = latent_dim
            self.in_channels = in_channels
            
            # #####
            # TODO: Complete the encoder architecture to calculate mu and log_var
            # mu and log_var will be used as inputs for the Reparameterization Trick,
            # generating latent vector z we need
            # #####
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.in_channels, 4, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(4),
                nn.ReLU(),
                nn.Conv2d(4, 8, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(8),
                nn.ReLU() 
            )

            self.conv2 = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(8, 12, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(12),
                nn.ReLU(),
                nn.Conv2d(12, 16, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(16),
                nn.ReLU() 
            )

            self.conv3 = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(16, 22, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(22),
                nn.ReLU(),
                nn.Conv2d(22, 28, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(28),
                nn.ReLU(),
                nn.Conv2d(28, 32, kernel_size = 2),
                nn.BatchNorm2d(32),
                nn.ReLU() 
            )
        
            self.fc_mean = torch.nn.Linear(1568, latent_dim, True)
            self.fc_log_var = torch.nn.Linear(1568, latent_dim, True)

        
        def forward(self, x):
            # #####
            # TODO: Complete the encoder architecture to calculate mu and log_var
            # #####
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x1 = x.view(x.size(0), -1)
            xmean = self.fc_mean(x1)
            xlogvar = self.fc_log_var(x1)

            return xmean, xlogvar


    # #####
    # TODO: Complete the decoder architecture
    # #####

    class Decoder(nn.Module):
        def __init__(
            self,
            latent_dim: int = 128,
            out_channels: int = 3,
            ):
            super(Decoder, self).__init__()
            self.latent_dim = latent_dim
            self.out_channels = out_channels
            
            # #####
            # TODO: Complete the decoder architecture to reconstruct image from latent vector z
            # #####
            self.para_linear = nn.Linear(latent_dim, 1568, True)

            self.upconv1 = nn.Sequential(
                nn.Conv2d(32, 28, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(28),
                nn.ReLU(),
                nn.Conv2d(28, 22, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(22),
                nn.ReLU(),
                nn.Conv2d(22, 16, kernel_size = 2, padding = 1),
                nn.BatchNorm2d(16),
                nn.ReLU()
            )
            
            self.upconv2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(16, 12, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(12),
                nn.ReLU(),
                nn.Conv2d(12, 8, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(8),
                nn.ReLU()
            )
            self.upconv3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(8, 4, kernel_size = 3, padding = 1), 
                nn.BatchNorm2d(4),
                nn.ReLU(),
                nn.Conv2d(4, out_channels=self.out_channels, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(self.out_channels),
                nn.Sigmoid()
            )

        def forward(self, z):
            # #####
            # TODO: Complete the decoder architecture to reconstruct image xg from latent vector z
            # #####
            if torch.cuda.is_available():
                z = z.cuda()
            
            z = self.para_linear(z)

            x = torch.reshape(z, (z.shape[0],32, 7, 7))
            
            x = self.upconv1(x)
            x = self.upconv2(x)
            x = self.upconv3(x)

            return x


    # #####
    # Wrapper for Variational Autoencoder
    # #####

    class VAE(nn.Module):
        def __init__(
            self, 
            latent_dim: int = 128,
            ):
            super(VAE, self).__init__()
            self.latent_dim = latent_dim

            self.encode = Encoder(latent_dim=latent_dim)
            self.decode = Decoder(latent_dim=latent_dim)

        def reparameterize(self, mu, log_var):
            """Reparameterization Tricks to sample latent vector z
            from distribution w/ mean and variance.
            """
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = eps * std + mu
            return z

        def forward(self, x, y = None):
            # #####
            # TODO: Complete forward for VAE
            # #####
            """Forward for CVAE.
            Returns:
                xg: reconstructed image from decoder.
                mu, log_var: mean and log(std) of z ~ N(mu, sigma^2)
                z: latent vector, z = mu + sigma * eps, acquired from reparameterization trick. 
            """
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x = self.decode(z)
            return x, mu, logvar, z

        def generate(
            self,
            n_samples: int,
            ):
            # #####
            # TODO: Complete generate method for VAE
            # #####

            """Randomly sample from the latent space and return
            the reconstructed samples.
            Returns:
                xg: reconstructed image
                None: a placeholder simply.
            """
            z = torch.randn(n_samples, self.latent_dim)
            x = self.decode(z)  
            return self.forward(x)[0], None


    # #####
    # Wrapper for Conditional Variational Autoencoder
    # #####

    class CVAE(nn.Module):
        def __init__(
            self, 
            latent_dim: int = 128,
            num_classes: int = 10,
            img_size: int = 32,
            ):
            super(CVAE, self).__init__()
            self.latent_dim = latent_dim
            self.num_classes = num_classes
            self.img_size = img_size

            # #####
            # TODO: Insert additional layers here to encode class information
            # Feel free to change parameters for encoder and decoder to suit your strategy
            # #####

            self.encode_class = nn.Linear(self.num_classes, self.img_size * self.img_size)
            self.encode_data = nn.Conv2d(3, 3, kernel_size=1)

            self.encode = Encoder(latent_dim=latent_dim, in_channels=3)
            self.cat_layer = nn.Linear(self.latent_dim + self.num_classes, self.latent_dim)
            self.decode = Decoder(latent_dim=latent_dim)



        def reparameterize(self, mu, log_var):
            """Reparameterization Tricks to sample latent vector z
            from distribution w/ mean and variance.
            """
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = eps * log_var + mu
            return z

        def forward(self, x, y):
            # #####
            # TODO: Complete forward for CVAE
            # Note that you need to process label information HERE.
            # #####
            """Forward for CVAE.
            Returns:
                xg: reconstructed image from decoder.
                mu, log_var: mean and log(std) of z ~ N(mu, sigma^2)
                z: latent vector, z = mu + sigma * eps, acquired from reparameterization trick. 
            """
            enc_class = self.encode_class(y)
            enc_class = enc_class.view(-1, self.image_size, self.image_size).unsqueeze(1)
            enc_data = self.encode_class(x)

            x = torch.cat([enc_data, enc_class], dim = 1)
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)

            xg = self.decode(z)
            return xg, mu, log_var, z


        def generate(
            self,
            n_samples: int,
            y: torch.Tensor = None,
            ):
            # #####
            # TODO: Complete generate for CVAE
            # #####
            """Randomly sample from the latent space and return
            the reconstructed samples.
            NOTE: Randomly generate some classes here, if not y is provided.
            Returns:
                xg: reconstructed image
                y: classes for xg. 
            """
            z = torch.randn(n_samples, self.latent_dim)
            z = z.to(device)

            z = torch.cat([z, y], dim = 1)
            xg = self.decode(z)
            return self.forward(xg)[0], y


    # #####
    # Wrapper for KL Divergence
    # #####

    class KLDivLoss(nn.Module):
        def __init__(
            self,
            lambd: float = 1.0,
            ):
            super(KLDivLoss, self).__init__()
            self.lambd = lambd

        def forward(
            self, 
            mu, 
            log_var,
            ):
            loss = 0.5 * torch.sum(-log_var - 1 + mu ** 2 + log_var.exp(), dim=1)
            return self.lambd * torch.mean(loss)


    # -----
    # Hyperparameters
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # NOTE: Feel free to change the hyperparameters as long as you meet the marking requirement
    # NOTE: DO NOT TRAIN IT LONGER THAN 100 EPOCHS.
    batch_size = 256
    workers = 2
    latent_dim = 128
    lr = 0.0005
    num_epochs = 50
    validate_every = 1
    print_every = 100

    conditional = True     # Flag to use VAE or CVAE

    if conditional:
        name = "cvae"
    else:
        name = "vae"

    # Set up save paths
    if not os.path.exists(os.path.join(os.path.curdir, "visualize", name)):
        os.makedirs(os.path.join(os.path.curdir, "visualize", name))
    save_path = os.path.join(os.path.curdir, "visualize", name)
    ckpt_path = name + '.pt'


    # TODO: Set up KL Annealing
    kl_annealing = [0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]      # KL Annealing


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

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True,
        transform=tfms,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=workers)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=workers)

    subset = torch.utils.data.Subset(
        test_dataset, 
        [0, 380, 500, 728, 1000, 2300, 3400, 4300, 4800, 5000])

    loader = torch.utils.data.DataLoader(
        subset, 
        batch_size=10)

    # -----
    # Model
    if conditional:
        model = CVAE(latent_dim=latent_dim)
    else:
        model = VAE(latent_dim=latent_dim)

    # -----
    # Losses
    # #####
    # TODO: Initialize your loss criterions HERE.
    # #####
    l2loss = nn.MSELoss()
    bceloss = nn.BCELoss()
    ssimloss = SSIM()
    klloss = KLDivLoss(kl_annealing[0])

    best_total_loss = float("inf")

    # Send to GPU
    if torch.cuda.is_available():
        model = model.cuda()



    optimizer = optim.Adam(model.parameters(), lr=lr)

    # To further help with training
    # NOTE: You can remove this if you find this unhelpful
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, [40, 50], gamma=0.1, verbose=False)


    # -----
    # Train loop

    # #####
    # TODO: Complete train_step for VAE/CVAE
    # #####

    def train_step(x, y): #remember to document
        optimizer.zero_grad() #setting optimizer gradient to zero to compute new gradients for backwards
        output,mean_coding,log_var_coding, _ = model(x, y) # run image and classes in case of cvae to get respective output
        loss1 = l2loss(output, x) #l2loss
        loss2 = bceloss(output, x) #bceloss
        loss3 = 1 - ssimloss(output, x) #ssimloss
        loss4 = klloss(mean_coding, log_var_coding) #kldivloss
        rcloss = loss1 + loss2 + loss3 #recon loss calculated
        tloss = rcloss + loss4 #total loss inclusive of all losses
        tloss.backward() #backwards to change the respective weights 
        optimizer.step()
        return loss1, rcloss, loss2, loss3, loss4

    def denormalize(x):
        """Denomalize a normalized image back to uint8.
        Args:
            x: torch.Tensor, in [0, 1].
        Return:
            x_denormalized: denormalized image as numpy.uint8, in [0, 255].
        """
        # #####
        # TODO: Complete denormalization.
        # #####
        x = torch.permute(x ,(0, 2, 3, 1))
        if device == "cuda:0":
            x = x.detach().cpu().numpy()
        else:
            x = x.detach().cpu().numpy()
        x = x * 255
        x_denormalized = x.round().astype(np.uint8)
        return x_denormalized

    # Loop HERE
    l2_losses = []
    bce_losses = []
    ssim_losses = []
    kld_losses = []
    total_losses = []

    total_losses_train = []

    for epoch in range(1, num_epochs + 1):
        total_loss_train = 0.0
        for i, (x, y) in enumerate(train_loader):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            # Train step
            model.train()
            l2_loss, recon_loss, bce_loss, ssim_loss, kldiv_loss = train_step(x, y)
            total_loss_train += recon_loss * x.shape[0]
            # Print
            if i % print_every == 0:
                print("Epoch {}, Iter {}: Total Loss: {:.6f} MSE: {:.6f}, SSIM: {:.6f}, BCE: {:.6f}, KLDiv: {:.6f}".format(epoch, i, recon_loss + kldiv_loss, l2_loss, ssim_loss, bce_loss, kldiv_loss))

        total_losses_train.append(total_loss_train.cpu().item() / len(train_dataset))

        # Test loop
        if epoch % validate_every == 0:
            # Loop through test set
            model.eval()

            # TODO: Accumulate average reconstruction losses per sample individually for plotting
            # Feel free to add code wherever you want to accumulate the loss


            with torch.no_grad():
                l_2loss = 0
                r_econloss = 0
                s_simloss = 0
                b_celoss = 0
                k_lloss = 0
                t_otalloss = 0
                for x, y in test_loader:
                    if torch.cuda.is_available():
                        x = x.cuda()
                        y = y.cuda()

                    output,mean_coding,log_var_coding, _ = model(x, y)
                    loss1 = l2loss(output, x) * x.shape[0]
                    loss2 = bceloss(output, x) * x.shape[0]
                    loss3 = (1 - ssimloss(output, x)) * x.shape[0]
                    loss4 = klloss(mean_coding, log_var_coding) * x.shape[0]
                    rcloss = loss1 + loss2 + loss3
                    # adding all the losses in order to get a net total loss for graphing
                    l_2loss += loss1.cpu()
                    b_celoss += loss2.cpu()
                    s_simloss += loss3.cpu()
                    k_lloss += loss4.cpu()
                    r_econloss += rcloss.cpu()
                    t_otalloss += rcloss.cpu() + loss4.cpu() 
                    # TODO: Accumulate average reconstruction losses per batch individually for plotting
                # dividing by the length of test/validation dataset to get my losses
                l2_losses.append(l_2loss.item()/ len(test_dataset))
                bce_losses.append(b_celoss.item()/ len(test_dataset))
                ssim_losses.append(s_simloss.item()/ len(test_dataset))
                kld_losses.append(k_lloss.item()/ len(test_dataset))
                total_losses.append(t_otalloss.item()/ len(test_dataset))
                avg_total_recon_loss_test = r_econloss.item() / (len(test_dataset))



                # Plot losses
                if epoch > 1:
                    plt.plot(l2_losses, label="L2 Reconstruction")
                    plt.plot(bce_losses, label="BCE")
                    plt.plot(ssim_losses, label="SSIM")
                    plt.plot(kld_losses, label="KL Divergence")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.xlim([1, epoch])
                    plt.legend()
                    plt.savefig(os.path.join(os.path.join(save_path, "losses.png")), dpi=300)
                    plt.clf()
                    plt.close('all')

                    plt.plot(total_losses, label="Total Loss Test")
                    plt.plot(total_losses_train, label="Total Loss Train")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.xlim([1, epoch])
                    plt.legend()
                    plt.savefig(os.path.join(os.path.join(save_path, "total_loss.png")), dpi=300)
                    plt.clf()
                    plt.close('all')
                
                # Save best model
                if avg_total_recon_loss_test < best_total_loss:
                    torch.save(model.state_dict(), ckpt_path)
                    best_total_loss = avg_total_recon_loss_test
                    print("Best model saved w/ Total Reconstruction Loss of {:.6f}.".format(best_total_loss))

            # Do some reconstruction
            model.eval()
            with torch.no_grad():
                x, y = next(iter(loader))
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                #y_onehot = F.one_hot(y, 10).float()
                xg, _, _, _ = model(x, y)

                # Visualize
                xg = denormalize(xg)
                x = denormalize(x)

                y = y.cpu().numpy()

                plt.figure(figsize=(10, 5))
                for p in range(10):
                    plt.subplot(4, 5, p+1)
                    plt.imshow(xg[p])
                    plt.subplot(4, 5, p + 1 + 10)
                    plt.imshow(x[p])
                    plt.text(0, 0, "{}".format(classes[y[p].item()]), color='black',
                                backgroundcolor='white', fontsize=8)
                    plt.axis('off')

                plt.savefig(os.path.join(os.path.join(save_path, "E{:d}.png".format(epoch))), dpi=300)
                plt.clf()
                plt.close('all')
                print("Figure saved at epoch {}.".format(epoch))

        # #####
        # TODO: Complete KL-Annealing.
        # #####
        # KL Annealing
        # Adjust scalar for KL Divergence loss
        klloss.lambd = kl_annealing[math.floor(epoch/10)]

        print("Lambda:", klloss.lambd)
        
        # LR decay
        scheduler.step()
        
        print()

    # Generate some random samples
    if conditional:
        model = CVAE(latent_dim=latent_dim)
    else:
        model = VAE(latent_dim=latent_dim)
    if torch.cuda.is_available():
        model = model.cuda()
    ckpt = torch.load(name+'.pt')
    model.load_state_dict(ckpt)

    # Generate 20 random images
    xg, y = model.generate(20)
    xg = denormalize(xg)
    if y is not None:
        y = y.cpu().numpy()

    plt.figure(figsize=(10, 5))
    for p in range(20):
        plt.subplot(4, 5, p+1)
        if y is not None:
            plt.text(0, 0, "{}".format(classes[y[p].item()]), color='black',
                    backgroundcolor='white', fontsize=8)
        plt.imshow(xg[p])
        plt.axis('off')

    plt.savefig(os.path.join(os.path.join(save_path, "random.png")), dpi=300)
    plt.clf()
    plt.close('all')

    if conditional:
        min_val, max_val = 0.92, 1
    else:
        min_val, max_val = 0.92, 1

    print("Total reconstruction loss:", best_total_loss)
    score = compute_score(best_total_loss, min_val, max_val)
    print("Your Assignment Score:", score)

if __name__ == '__main__':
    main()