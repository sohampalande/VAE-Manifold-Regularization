import torch
import torchvision
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt; 
from utils import Visualize as viz
from model import VariationalAutoencoder 
from utils import MNISTSampler



if __name__ == '__main__':

    ######### Load Datasets ################

    #load transfer dataset
    mnist = datasets.MNIST(root='./data/', train=True, download=False, transform=torchvision.transforms.ToTensor())

    mask = [1 if mnist[i][1] == 0 or mnist[i][1] == 9 else 0 for i in range(len(mnist))]   #change digits as per requirement of dataset
    mask = torch.tensor(mask)   
    sampler = MNISTSampler(mask, mnist)
    trainloader_transfer = torch.utils.data.DataLoader(mnist, batch_size=128, sampler = sampler, shuffle=False)

    #load original dataset
    mask = [1 if mnist[i][1] == 6 or mnist[i][1] == 4 else 0 for i in range(len(mnist))]
    mask = torch.tensor(mask)   
    sampler = MNISTSampler(mask, mnist)
    trainloader_original = torch.utils.data.DataLoader(mnist, batch_size=128, sampler = sampler, shuffle=False)

    #load joint dataset
    mask = [1 if mnist[i][1] == 0 or mnist[i][1] == 9 or mnist[i][1] == 6 or mnist[i][1] == 4  else 0 for i in range(len(mnist))]  #change digits as per requirement of dataset
    mask = torch.tensor(mask)   
    sampler = MNISTSampler(mask, mnist)
    trainloader_joint = torch.utils.data.DataLoader(mnist, batch_size=128, sampler = sampler, shuffle=False)

    ######## Train Model  #################

    epochs=220      #set number of epochs
    lr = 0.00001
    experiment_directory = '0-9_6-4'   #name of experiment folder
    save_video=True #change to True if you want the model to create a video of the latent space through learning
    path = './MNIST_results/' + experiment_directory + '/models/vae_model_2.pt'

    vae_model_2 = torch.load(path)

    reconstruction_space_path = "./MNIST_results/0-9_6-4/models/transfer_learning_latent_space/reconstruction2_space_"
    latent_space_path = "./MNIST_results/0-9_6-4/models/transfer_learning_latent_space/latent2_space_"
    synthetic_data_path = "./MNIST_results/0-9_6-4/models/transfer_learning_latent_space/syn2_data_"

    if not save_video:
        synthetic_data_path=""

    vae_model_2_transfer = vae_model_2.train_transfer(data=trainloader_transfer, joint_data=trainloader_joint, reconstruction_space_path=reconstruction_space_path, latent_space_path=latent_space_path, synthetic_data_path=synthetic_data_path, epochs=epochs, lr=lr)

    path = './MNIST_results/' + experiment_directory + '/models/vae_model_2_transfer.pt'
    torch.save(vae_model_2_transfer, path) #save model

    reconstruction_space_path = reconstruction_space_path + "%03d.png"
    latent_space_path = latent_space_path + "%03d.png"
    synthetic_data_path = synthetic_data_path + "%03d.png"

    if not save_video:
        synthetic_data_path=""

    video_path_recons = './MNIST_results/0-9_6-4/models/transfer_learning_videos/reconstruction2_space.avi'
    video_path_latent = './MNIST_results/0-9_6-4/models/transfer_learning_videos/latent2_space.avi'
    video_path_synthetic = './MNIST_results/0-9_6-4/models/transfer_learning_videos/synthetic2_data.avi'

    fps_1 = 15
    fps_2 = 10
    fps_3 = 15

    viz.create_video(reconstruction_space_path, video_filepath=video_path_recons, fps=fps_1)
    viz.create_video(latent_space_path, video_filepath=video_path_latent, fps=fps_2)
    viz.create_video(synthetic_data_path, video_filepath=video_path_synthetic, fps=fps_3)


    ############ Visualize #################

    viz.plot_reconstructions_normal(vae_model_2_transfer)

    # plot t-SNE

    #get dataset and samples
    n_tsne = 3500

    dataset1 = viz.get_dataset_from_loader(trainloader_original)
    print(dataset1.shape)
    idx = np.random.choice(dataset1.shape[0], n_tsne, replace=False)
    dataset1 = dataset1[idx]

    dataset2 = viz.get_dataset_from_loader(trainloader_transfer)
    print(dataset2.shape)
    idx = np.random.choice(dataset2.shape[0], n_tsne, replace=False)
    dataset2 = dataset2[idx]


    samples = viz.generate_samples(vae_model_2_transfer, n_tsne)

    #reshape dataset and samples
    dataset1 = np.reshape(dataset1, newshape=(n_tsne, 28*28))
    dataset2 = np.reshape(dataset2, newshape=(n_tsne, 28*28))
    samples = np.reshape(samples, newshape=(n_tsne, 28*28))

    viz.get_TSNE_transfer(dataset1, dataset2, samples)

    plt.show()

    