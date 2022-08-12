from timevae import TimeVAE
from utils import Visualize as viz
from utils import MinMaxScaler
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
    dataset_name =  'AMZN_GME_synthetic'         #AMZN_10-20, GME_10-20, AMZN_GME, AMZN_GME_synthetic, NFLX
    file_path = './datasets/' + dataset_name + '_preprocessed.npy'


    #load dataset
    dataset = np.load(file_path)
    print(dataset)
    print("The shape of the dataset is: ", dataset.shape)

    #set parameters
    seq_len = dataset.shape[-1]
    feat_dim = dataset.shape[1]    
    latent_dim = 8
    hidden_layer_sizes = [100, 250]
    reconstruction_wt = 3
    kernel_size = 3
    seed = 1    # 1, 2, 3, 4, 5
    set_seed(seed)
    val_set = 0.1
    epochs = 55
    batch_size = 32
    lr=0.001

    # #save path
    folder = 'AMZN_GME'   #AMZN, GME, NFLX, oracle, AMZN_GME, GME_AMZN
    save_path = "./models/" + folder + "/" + "seed_" + str(seed)
    save_path=''     #uncomment this line if you do not wish to save training model

    #initialize and train
    time_vae = TimeVAE(seq_len=seq_len, feat_dim=feat_dim, latent_dim=latent_dim, hidden_layer_sizes=hidden_layer_sizes, 
                    reconstruction_wt=reconstruction_wt, kernel_size=kernel_size)

    summary(time_vae, (1, dataset.shape[1], dataset.shape[-1]))

    model = time_vae.fit(dataset=dataset, dataset_name=dataset_name, seed=seed, batch_size=batch_size, lr=lr, val_set=val_set, file_path=save_path, epochs=epochs)

    print("Model Trained.")

    ##################### Plots ################################################################

    scaler = model.scaler #set model scaler
    path = save_path      #set path to save plots
    fontsize=14

    ########### plot training data and reconstructions ###################
    #scale and convert to tensor
    scaled_data = scaler.fit_transform(dataset)
    scaled_data = torch.Tensor(scaled_data)

    # #forward pass through model
    encoder_output, reconstructions, z_log_var, z_mean = model(scaled_data)

    viz.draw_orig_and_post_pred_sample_timeseries(orig=scaled_data, reconst=reconstructions, n=5, path=path)
    viz.draw_orig_and_post_pred_sample(orig=scaled_data, reconst=reconstructions, n=5, path="") #not saving images

 
    ########## Generate Random Samples ###################################
    #model = torch.load('./models/AMZN_GME/seed_1/model.pt')
    num_samples = 75 #generate 75 random samples
    viz.plot_samples(model=model, n=num_samples, inverse_transform=True, path=path)

    #plot samples from training dataset
    path = './models/AMZN_GME/seed_1/training_data.png'
    plt.figure()
    plt.title('Training Data', fontsize=fontsize)
    ind = np.random.rand(num_samples)*dataset.shape[0]
    for i in ind:
        plt.plot(dataset[int(i), 0, :].T)
    plt.savefig(path)
    plt.show()
    
    ########### T-SNE Plots ##############################################
    num_samples = dataset.shape[0]
    samples = model.get_prior_samples(num_samples=num_samples)

    dataset_returns = []
    samples_returns = []

    for window in dataset:
        window = np.squeeze(window)
        window_returns = np.log(window[1:]) - np.log(window[:-1])
        dataset_returns.append(window_returns)
    
    samples = scaler.inverse_transform(samples)

    for window in samples:
        window = np.squeeze(window)
        window_returns = np.log(window[1:]) - np.log(window[:-1])
        samples_returns.append(window_returns)
    
    dataset_returns = np.array(dataset_returns)
    samples_returns = np.array(samples_returns)

    plt.figure()
    plt.title("Returns Distribution")
    plt.hist(dataset_returns.flatten(), color='red', alpha=0.5, density=True, bins=10)
    plt.hist(samples_returns.flatten(), color='blue', alpha=0.5, density=True, bins=10)

    viz.get_TSNE(dataset=dataset_returns, samples=samples_returns, path=path)
    viz.get_TSNE(dataset=dataset, samples=samples, path=path)


    # # t-SNE visualize means 
    num_samples = dataset.shape[0]
    samples = model.get_prior_samples(num_samples=num_samples)
    samples = torch.tensor(samples)

    encoder_output_samples, reconstructions_samples, z_log_var_samples, z_mean_samples = model(samples)
    encoder_output_dataset, reconstructions_dataset, z_log_var_dataset, z_mean_dataset = model(scaled_data)

    z_mean_samples = z_mean_samples.detach().numpy()
    z_mean_dataset = z_mean_dataset.detach().numpy()

    viz.get_TSNE(dataset=z_mean_dataset, samples=z_mean_samples, path=path, means=True)


    plt.show()

