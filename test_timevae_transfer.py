from cProfile import label
from turtle import color
from timevae import TimeVAE
from utils import Visualize as viz
import torch
import numpy as np
from torchinfo import summary
import matplotlib
import matplotlib.pyplot as plt;


def set_seed(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':

    #set path
    dataset_name =  'AMZN_GME_synthetic'         #AMZN_GME_synthetic
    file_path = './datasets/' + dataset_name + '_preprocessed.npy'

    #load dataset
    dataset = np.load(file_path)
    print("The shape of the dataset is: ", dataset.shape)

    #set parameters
    seed = 5    # 1, 2, 3, 4, 5
    set_seed(seed)
    val_set = 0.1
    epochs = 10
    batch_size = 32
    lr=0.001

    # #save path
    folder = 'GME_AMZN'   #AMZN_GME, GME_AMZN
    save_path = "./models/" + folder + "/" + "seed_" + str(seed) 
    save_path=''

    #initialize and train
    #load base model with specified seed
    base_model = "GME"   #AMZN, GME
    model_path = "./models/" + base_model + "/" + "seed_" + str(seed) + "/model.pt"
    model = torch.load(model_path) #load the specified seed of the model

    summary(model, (1, dataset.shape[1], dataset.shape[-1]))

    model = model.fit(dataset=dataset, dataset_name=dataset_name, seed=seed, batch_size=batch_size, lr=lr, val_set=val_set, file_path=save_path, epochs=epochs)

    print("Model Trained.")

    ##################### Plots ################################################################

    scaler = model.scaler #load model scaler
    path = save_path      #set path to save plots

    ########### plot training data and reconstructions ###################
    #scale and convert to tensor
    scaled_data = scaler.fit_transform(dataset)
    scaled_data = torch.Tensor(scaled_data)

    #forward pass through model
    encoder_output, reconstructions, z_log_var, z_mean = model(scaled_data)

    viz.draw_orig_and_post_pred_sample_timeseries(orig=scaled_data, reconst=reconstructions, n=5, path=path)
    viz.draw_orig_and_post_pred_sample(orig=scaled_data, reconst=reconstructions, n=5, path="")


    ########## Generate Random Samples ###################################
    num_samples = 75 #generate 75 random samples

    viz.plot_samples(model=model, n=num_samples, inverse_transform=True, path=path)


    ########### T-SNE Plots ##############################################
    # Load the component datasets of the training dataset
    dataset1_name =  'AMZN_synthetic'         #AMZN_GME_synthetic
    file_path1 = './datasets/' + dataset_name + '_preprocessed.npy'

    dataset2_name =  'GME_synthetic'         #AMZN_GME_synthetic
    file_path2 = './datasets/' + dataset_name + '_preprocessed.npy'

    dataset1 = np.load(file_path1)
    dataset2 = np.load(file_path2)

    scaled_data1 = scaler.fit_transform(dataset1)  #scale the datasets
    scaled_data1 = torch.Tensor(scaled_data1)

    scaled_data2 = scaler.fit_transform(dataset2)
    scaled_data2 = torch.Tensor(scaled_data2)

    num_samples = scaled_data1.shape[0]*2
    samples = model.get_prior_samples(num_samples=num_samples)

    #plot the scaled datasets 
    plt.figure()

    plt.plot(scaled_data1[:, 0].T, label="AMZN", color='red')
    plt.plot(scaled_data2[:,0].T, label='GME', color='blue', alpha=0.1)
    plt.title('Scaled AMZN and GME Training Data')

    viz.get_TSNE_transfer(dataset1=scaled_data1, dataset2 = scaled_data2, samples=samples, path=path)

    # T-SNE visualize means 
    num_samples = scaled_data1.shape[0]*2
    samples = model.get_prior_samples(num_samples=num_samples)
    samples = torch.tensor(samples)

    encoder_output_samples, reconstructions_samples, z_log_var_samples, z_mean_samples = model(samples)
    encoder_output_dataset1, reconstructions_dataset1, z_log_var_dataset1, z_mean_dataset1 = model(scaled_data1)
    encoder_output_dataset2, reconstructions_dataset2, z_log_var_dataset2, z_mean_dataset2 = model(scaled_data2)

    z_mean_samples = z_mean_samples.detach().numpy()
    z_mean_dataset1 = z_mean_dataset1.detach().numpy()
    z_mean_dataset2 = z_mean_dataset2.detach().numpy()

    viz.get_TSNE_transfer(dataset1=z_mean_dataset1, dataset2=z_mean_dataset2, samples=z_mean_samples, means=True, path=path)

    # T-SNE visualize 2-D slices of latent space

    encoder_output_dataset1 = encoder_output_dataset1.detach().numpy()    #convert to np arrays
    encoder_output_dataset2 = encoder_output_dataset2.detach().numpy()
    embeddings_real = np.append(encoder_output_dataset1, encoder_output_dataset2, axis=0) #append the two datasets

    embeddings_fake = encoder_output_samples.detach().numpy()

    viz.plot_slices(embeddings_real=embeddings_real, 
                    embeddings_fake=embeddings_fake, path=path)

    plt.show()
