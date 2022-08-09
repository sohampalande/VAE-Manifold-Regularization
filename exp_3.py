import torch
import torchvision
import torchvision.datasets as datasets
import numpy as np
from utils import Visualize as viz
from model import VariationalAutoencoder 
from utils import MNISTSampler
import matplotlib.pyplot as plt; 




if __name__ == '__main__':

    ###### create MNIST Dataset ##############

    #data for model 3
    # Dataset 3
    mnist = datasets.MNIST(root='./data/', train=True, download=False, transform=torchvision.transforms.ToTensor())

    mask = [1 if mnist[i][1] == 0 or mnist[i][1] == 9 or mnist[i][1] == 6 or mnist[i][1] == 4  else 0 for i in range(len(mnist))]  #change digits as per requirement of dataset
    mask = torch.tensor(mask)   
    sampler = MNISTSampler(mask, mnist)
    trainloader = torch.utils.data.DataLoader(mnist, batch_size=128, sampler = sampler, shuffle=False)

    images, labels = next(iter(trainloader))
    print("Shape of Images: ", images.shape)
    viz.imshow(images[0], normalize=False)

    ######## Train Model ##################

    #set hyperparamters
    latent_dim = 2
    epochs = 175
    directory = "0-9_6-4"   #change directory to dataset folder
    model_name = 'vae_oracle_model' #change name of model

    path = './MNIST_results/' + directory + '/models/' +  model_name 

    vae_oracle_model = VariationalAutoencoder(latent_dim)
    vae_oracle_model = vae_oracle_model.train(trainloader, epochs=epochs)

    torch.save(vae_oracle_model, path)

    ########### Visualize ###############

    viz.plot_latent(vae_oracle_model, trainloader)  #latent space
    viz.plot_reconstructions_space(vae_oracle_model)  #latent space reconstructions

    viz.plot_reconstructions_normal(vae_oracle_model)   #synthetic data

    #t-SNE

    #get dataset and samples
    n_tsne = 3500

    dataset1 = viz.get_dataset_from_loader(trainloader)
    idx = np.random.choice(dataset1.shape[0], n_tsne, replace=False)
    dataset1 = dataset1[idx]
    samples = viz.generate_samples(vae_oracle_model, n_tsne)

    #reshape dataset and samples
    dataset1 = np.reshape(dataset1, newshape=(n_tsne, 28*28))
    samples = np.reshape(samples, newshape=(n_tsne, 28*28))

    viz.get_TSNE(dataset1, samples)

    plt.show()