import torch
import torchvision
import torchvision.datasets as datasets
import numpy as np
from utils import Visualize as viz
from model import VariationalAutoencoder 
from utils import MNISTSampler
import matplotlib.pyplot as plt 





if __name__ == '__main__':

    ###### create MNIST Datasets ##############

    #data for model 1
    # Dataset 1
    mnist = datasets.MNIST(root='./data/', train=True, download=False, transform=torchvision.transforms.ToTensor())

    mask = [1 if mnist[i][1] == 0 or mnist[i][1] == 9 else 0 for i in range(len(mnist))]
    mask = torch.tensor(mask)   
    sampler = MNISTSampler(mask, mnist)
    trainloader = torch.utils.data.DataLoader(mnist, batch_size=128, sampler = sampler, shuffle=False)

    images, labels = next(iter(trainloader))
    print("Shape of Images: ", images.shape)
    viz.imshow(images[0], normalize=False)


    ######## Train Model ##################

    #set hyperparamters
    latent_dim = 2
    epochs = 150
    directory = "0-9_6-4"   #change directory to dataset folder
    model_name = 'vae_model_1' #change name of model

    path = './MNIST_results/' + directory + '/models/' +  model_name 

    vae_model_1 = VariationalAutoencoder(latent_dim)
    vae_model_1 = vae_model_1.train(trainloader, epochs=epochs)

    torch.save(vae_model_1, path)

    ########### Visualize ###############

    viz.plot_latent(vae_model_1, trainloader)  #latent space
    viz.plot_reconstructions_space(vae_model_1)  #latent space reconstructions

    viz.plot_reconstructions_normal(vae_model_1)   #synthetic data

    #t-SNE

    #get dataset and samples
    n_tsne = 3500

    dataset1 = viz.get_dataset_from_loader(trainloader)
    idx = np.random.choice(dataset1.shape[0], n_tsne, replace=False)
    dataset1 = dataset1[idx]
    samples = viz.generate_samples(vae_model_1, n_tsne)

    #reshape dataset and samples
    dataset1 = np.reshape(dataset1, newshape=(n_tsne, 28*28))
    samples = np.reshape(samples, newshape=(n_tsne, 28*28))

    viz.get_TSNE(dataset1, samples)

    plt.show()




