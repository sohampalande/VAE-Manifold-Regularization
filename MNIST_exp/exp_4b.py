import torch
import torchvision
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt; 
from utils import Visualize as viz
from model import VariationalAutoencoder 
from utils import MNISTSampler



if __name__ == '__main__':

    ############# create dataset: without regularization ##################
    
    n = 5000  #size of synthetic dataset
    experiment_directory = '0-9_6-4'   #name of experiment folder

    #load models
    path_1 = './MNIST_results/' + experiment_directory + '/models/vae_model_1'
    path_2 = './MNIST_results/' + experiment_directory + '/models/vae_model_2'
    
    vae_model_1 = torch.load(path_1)
    vae_model_2 = torch.load(path_2)

    samples1 = torch.Tensor(viz.generate_samples(vae_model_1, n))
    samples2 = torch.Tensor(viz.generate_samples(vae_model_2, n))

    print(samples1.shape)

    samples_dataset = []

    for sample in samples1:
        samples_dataset.append([sample.view(-1, 28, 28), 0])
    
    for sample in samples2:
        samples_dataset.append([sample.view(-1, 28, 28), 1])

    #create batch_sizes
    batch_size = 128
    dataloader_synthetic = torch.utils.data.DataLoader(samples_dataset, batch_size=128, shuffle=True)

    
    ############ Train Model ################

    #train model for joint model trained on synthetic data
    latent_dim = 2
    epochs = 150
    path = '/content/drive/MyDrive/JPMC_summer2022/image_experiments_output/exp_4/no_regularization/vae_model_joint_synthetic'

    vae_model_joint_synthetic = VariationalAutoencoder(latent_dim)
    vae_model_joint_synthetic = vae_model_joint_synthetic.train(dataloader_synthetic, epochs=epochs)

    #save model
    path = './MNIST_results/' + experiment_directory + '/models/vae_model_joint_synthetic'
    torch.save(vae_model_joint_synthetic, path)


    ########### Visualize ###################

    #create joint dataset - use original since it is labelled
    mnist = datasets.MNIST(root='./data/', train=True, download=False, transform=torchvision.transforms.ToTensor())

    mask = [1 if mnist[i][1] == 0 or mnist[i][1] == 9 or mnist[i][1] == 6 or mnist[i][1] == 4  else 0 for i in range(len(mnist))]  #change digits as per requirement of dataset
    mask = torch.tensor(mask)   
    sampler = MNISTSampler(mask, mnist)
    trainloader = torch.utils.data.DataLoader(mnist, batch_size=128, sampler = sampler, shuffle=False)


    viz.plot_latent(vae_model_joint_synthetic, trainloader)
    viz.plot_latent(vae_model_joint_synthetic, dataloader_synthetic)
    viz.plot_reconstructions_space(vae_model_joint_synthetic)

    viz.plot_reconstructions_normal(vae_model_joint_synthetic)

    # plot t-SNE

    #get dataset and samples
    dataset1 = np.copy(samples1)
    dataset2 = np.copy(samples2)
    samples = viz.generate_samples(vae_model_joint_synthetic, n=dataset1.shape[0]+dataset2.shape[0])

    #reshape dataset and samples
    dataset1 = np.reshape(dataset1, newshape=(dataset1.shape[0], 28*28))
    dataset2 = np.reshape(dataset2, newshape=(dataset2.shape[0], 28*28))
    samples = np.reshape(samples, newshape=(samples.shape[0], 28*28))

    viz.get_TSNE_transfer(dataset1=dataset1, dataset2=dataset2,  samples=samples)

    plt.show()

    