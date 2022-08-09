import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt; 
#plt.rcParams['figure.dpi'] = 150
import random
import seaborn as sns
import cv2

from utils import Visualize as viz

#define a class for the encoder

class Encoder(torch.nn.Module):

    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        #define the model architecture

        self.encoder1 = torch.nn.Linear(in_features=784, out_features=512)
        self.encoder2 = torch.nn.Linear(in_features=512, out_features=latent_dim)
        self.encoder3 = torch.nn.Linear(512, latent_dim)

        #define target distribution 
        self.N = torch.distributions.Normal(0,1) #reutrns the location (mean) and scale (variance)
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.encoder1(x)) #pass input through first layer of encoder and apply the relu activation function
        
        mu = self.encoder2(x)
        sigma = torch.exp(self.encoder3(x))

        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, sigma


class Decoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.decoder1 = torch.nn.Linear(latent_dim, 512)
        self.decoder2 = torch.nn.Linear(512, 784)

    def forward(self, x):
        x = F.relu(self.decoder1(x))
        x = torch.sigmoid(self.decoder2(x))
        return x.reshape((-1, 1, 28, 28))



#combine encoder and decoder models in a single model
class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super(VariationalAutoencoder, self).__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim


    def forward(self, x):
        x, mu, sigma = self.encoder(x)
        x = self.decoder(x)
        return x


    def train(self, data, epochs=20, lr=0.001):

        vae_model = self

        opt = torch.optim.Adam(vae_model.parameters(), lr=lr) #use the adam optimizer
        losses = []
        
        for epoch in range(epochs):
            print("Epoch: ", epoch)
            
            for x,y in data:    
                opt.zero_grad() #sets the gradient for each param to 0 
                x_hat = vae_model(x) #uses the forward method defined in parent class
                loss = ((x-x_hat)**2).sum() + vae_model.encoder.kl
                loss.backward() #compute dloss/dx w.r.t. every param x in the model // accumulates the gradient by addition when called multiple times
                opt.step() #update parameters based on current gradient and update rule
                
            losses.append(loss.detach().numpy())
            print("Loss: ", loss.detach().numpy())
            print()
        
        #plot total losses
        plt.plot(losses)
        plt.title('Model Loss', fontsize = 14)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()
        
        return vae_model


    def train_transfer(self, data, joint_data, reconstruction_space_path="", latent_space_path="", synthetic_data_path="", epochs=20, lr=0.001):

        vae_model = self

        opt = torch.optim.Adam(vae_model.parameters(), lr=lr) #use the adam optimizer
        
        save_codes = ["%03d" % i for i in range(epochs)]

        losses = []
        
        for epoch in range(epochs):
            print("Epoch: ", epoch)

            #generate and save plots for each epoch
            if reconstruction_space_path!="" and latent_space_path!="" and synthetic_data_path!="":
                recons_space_path = reconstruction_space_path + save_codes[epoch] + ".png"
                lat_space_path = latent_space_path + save_codes[epoch] + ".png"
                syn_data_path = synthetic_data_path + save_codes[epoch] + ".png"
            
                viz.plot_reconstructions_space(vae_model, path=recons_space_path, show_plots=False)
                viz.plot_latent(vae_model, joint_data, path=lat_space_path, show_plots=False)
                viz.plot_reconstructions_normal(vae_model, path=syn_data_path, show_plots=False)
        
            for x,y in data:    
                opt.zero_grad() #sets the gradient for each param to 0 
                x_hat = vae_model(x) #uses the forward method defined in parent class
                loss = ((x-x_hat)**2).sum() + vae_model.encoder.kl
                loss.backward() #compute dloss/dx w.r.t. every param x in the model // accumulates the gradient by addition when called multiple times
                opt.step() #update parameters based on current gradient and update rule
            
            losses.append(loss.detach().numpy())
            print("Loss: ", loss.detach().numpy())
            print()
        
        #plot total losses
        plt.plot(losses)
        plt.title('Model Loss', fontsize = 14)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

        return vae_model

    
    def train_manifold_regularization(self, data, laplacians, lambd=1, epochs=20, lr=0.001):

        vae_model = self

        opt = torch.optim.Adam(vae_model.parameters(), lr=lr) #use the adam optimizer
        losses = []
        reg_losses = []
        
        for epoch in range(epochs):
            
            for x, model_idx, sample_num in data:    
                opt.zero_grad() #sets the gradient for each param to 0 
                x_hat = vae_model(x) #uses the forward method defined in parent class
                loss = ((x-x_hat)**2).sum() + vae_model.encoder.kl

                reg_loss = 0

                model_idx = model_idx.detach().numpy()
                sample_num = sample_num.detach().numpy()
            
                for i in range(len(laplacians)):

                    L = torch.tensor(laplacians[i])
                    mask = (model_idx==i)       #index by model number
                    mask_l = torch.tensor(sample_num[mask])   #index samples list by model number
                    L = torch.index_select(L, 0, mask_l)
                    L = torch.index_select(L, 1, mask_l).float()

                    embeddings, mu, sigma = vae_model.encoder(x[mask])

                    for d in range(vae_model.latent_dim):
                        reg_loss += torch.matmul(torch.matmul(embeddings[:,d].T, L), embeddings[:,d])
                

                loss += lambd*reg_loss

                loss.backward() #compute dloss/dx w.r.t. every param x in the model // accumulates the gradient by addition when called multiple times
                opt.step() #update parameters based on current gradient and update rule
            
            losses.append(loss.detach().numpy())
            reg_losses.append(reg_loss.detach().numpy())
            print(f"Epoch {epoch} complete: Total loss = {losses[-1]: .4f}, Energy loss = {reg_losses[-1]: .4f}")
            print()

        
        #plot total losses
        plt.plot(losses, label='Total Loss')
        plt.plot(reg_losses, label='Diriclet Energy Loss')
        plt.title('Model Loss', fontsize = 14)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
        
        return vae_model


