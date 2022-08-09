import torch
import torchvision
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt; 
from utils import Visualize as viz
from model import VariationalAutoencoder 
from utils import MNISTSampler

from utils import GraphConstruction as graph





if __name__ == '__main__':


    create_new_embedding = False #change to True if you want a new embedding matrix
    samples = 3000               # number of synthetic data points to use for embedding matrix 
    experiment_directory = '0-9_6-4'   #name of experiment folder

    if create_new_embedding:

        #create synthetic dataset - with manifold regularization

        #get synthetic data and weight matrix for samples generated from both models
        path_1 = './MNIST_results/' + experiment_directory + '/models/vae_model_1'
        path_2 = './MNIST_results/' + experiment_directory + '/models/vae_model_2'

        model_1 = torch.load(path_1)
        W_1_base, synthetic_data_1 = graph.get_embedding_matrix(model=model_1, samples=samples, gamma=1)

        model_2 = torch.load(path_2)
        W_2_base, synthetic_data_2 = graph.get_embedding_matrix(model=model_2, samples=samples, gamma=1)

        #save graphs and synthetic data
        path = './MNIST_results/' + experiment_directory + '/models/graphs/W_1_base.npy'
        np.save(path, W_1_base)

        path = './MNIST_results/' + experiment_directory + '/models/graphs/W_2_base.npy'
        np.save(path, W_2_base)

        path = './MNIST_results/' + experiment_directory + '/models/graphs/synthetic_data_1.npy'
        np.save(path, synthetic_data_1)

        path = './MNIST_results/' + experiment_directory + '/models/graphs/synthetic_data_2.npy'
        np.save(path, synthetic_data_2)
    
    else: 

        #load synthetic data and correponding graphs

        path_1 = './MNIST_results/' + experiment_directory + '/models/vae_model_1'
        path_2 = './MNIST_results/' + experiment_directory + '/models/vae_model_2'

        W_1_base = np.load(path_1)
        W_2_base = np.load(path_2)

        path_syn_1 = './MNIST_results/' + experiment_directory + '/models/graphs/synthetic_data_1.npy'
        path_syn_2 = './MNIST_results/' + experiment_directory + '/models/graphs/synthetic_data_2.npy'
        
        synthetic_data_1 = np.load(path_syn_1)
        synthetic_data_2 = np.load(path_syn_2)
                           

        #trasnform weight matrix
        gamma = 8       

        W_1 = np.exp(gamma*np.log(W_1_base))
        W_2 = np.exp(gamma*np.log(W_2_base))

        #combine synthetic datasets  
        #each entry in synthetic_dataset is formatted [sample, model_idx, sample_num]

        synthetic_dataset = []

        i=0
        for sample in synthetic_data_1:
            synthetic_dataset.append([sample, 0, i])
            i+=1

        i=0
        for sample in synthetic_data_2:
            synthetic_dataset.append([sample, 1, i])
            i+=1

        #get Laplacians corresponding to the weight matrices for synthetic_data_1 and synthetic_data_2
        L_1 = graph.get_laplacian(W_1)
        L_2 = graph.get_laplacian(W_2)

        laplacians = [L_1, L_2]

        #create dataloader object and plot a sample 
        dataloader_synthetic = torch.utils.data.DataLoader(synthetic_dataset, batch_size=128, shuffle=True)

        samples, model_idx, sample_num = next(iter(dataloader_synthetic))
        viz.imshow(samples[0], normalize=False)
    

    latent_dim = 2
    epochs = 300
    path = '/content/drive/MyDrive/JPMC_summer2022/image_experiments_output/exp_4/regularization/vae_model_joint_synthetic_reg'
    lambd = 2

    vae_model_joint_synthetic_reg = VariationalAutoencoder(latent_dim)
    vae_model_joint_synthetic_reg = vae_model_joint_synthetic_reg.train_manifold_regularization(dataloader_synthetic, laplacians=laplacians, epochs=epochs, lambd=lambd)

    #save model
    path = './MNIST_results/' + experiment_directory + '/models/vae_model_joint_synthetic_reg'
    torch.save(vae_model_joint_synthetic_reg, path)


    ########### Visualizations ###################

    #create joint dataset - use original since it is labelled
    mnist = datasets.MNIST(root='./data/', train=True, download=False, transform=torchvision.transforms.ToTensor())

    mask = [1 if mnist[i][1] == 0 or mnist[i][1] == 9 or mnist[i][1] == 6 or mnist[i][1] == 4  else 0 for i in range(len(mnist))]  #change digits as per requirement of dataset
    mask = torch.tensor(mask)   
    sampler = MNISTSampler(mask, mnist)
    trainloader = torch.utils.data.DataLoader(mnist, batch_size=128, sampler = sampler, shuffle=False)

    path = './MNIST_results/' + experiment_directory + '/models/vae_model_joint_synthetic_reg'
    model = torch.load(path)

    viz.plot_latent(vae_model_joint_synthetic_reg, trainloader, lim=[-2,2])  #set lim to adjust viewing window
    viz.plot_latent_regularization(model, dataloader_synthetic, lim=[-2,2])

    viz.plot_reconstructions_space(vae_model_joint_synthetic_reg)
    viz.plot_reconstructions_normal(vae_model_joint_synthetic_reg)


    #plot t-SNE

    #get dataset and samples
    dataset1 = synthetic_data_1
    dataset2 = synthetic_data_2
    samples = viz.generate_samples(vae_model_joint_synthetic_reg, n=dataset1.shape[0]+dataset2.shape[0])

    #reshape dataset and samples
    dataset1 = np.reshape(dataset1, newshape=(dataset1.shape[0], 28*28))
    dataset2 = np.reshape(dataset2, newshape=(dataset2.shape[0], 28*28))
    samples = np.reshape(samples, newshape=(samples.shape[0], 28*28))

    viz.get_TSNE_transfer(dataset1=dataset1, dataset2=dataset2,  samples=samples)

    plt.show()