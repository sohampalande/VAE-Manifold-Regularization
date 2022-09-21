import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.tsvae_conv import ConvTimeSeriesVAE
from models.tsvae_lstm import LSTMTimeSeriesVAE

# Global parameters
features = ['Open', 'High', 'Low', 'Close', 'Adj Close']
time_name = 'Date'
dataset_name = 'GME_10-20'
latent_dimension = 20
hidden_layers = [50, 100, 200]
hidden_size = 60*len(features)
reconstruction_wt = 1
kernel_size = 4
epochs = 50
batch_size = 64
num_layers = 1
lr = 0.001
seq_len = 100


if __name__ == '__main__':
    # Load dataset
    file_path = './datasets/' + dataset_name + '.csv'
    dataset = pd.read_csv(file_path)

    # Instantiate model and train
    model = LSTMTimeSeriesVAE(seq_len=seq_len, dataset=dataset, time_column=time_name, feature_names=features,
                              latent_dim=latent_dimension, hidden_size=hidden_size,
                              reconstruction_wt=reconstruction_wt, num_layers=num_layers, segment_length=seq_len)
    # model = ConvTimeSeriesVAE(seq_len=seq_len, dataset=dataset, time_column=time_name, feature_names=features,
    #                           latent_dim=latent_dimension, hidden_layer_sizes=hidden_layers,
    #                           reconstruction_wt=reconstruction_wt, kernel_size=kernel_size)
    model.fit(batch_size=batch_size, lr=lr, epochs=epochs)


    # Test to see quality of generated synthetic data
    N = 100
    samples = model.sample(N, return_dataframe=False)
    compare_idx = np.random.choice(model.dataset.shape[0], N, replace=False)
    for i in range(samples.shape[2]):
        plt.figure()
        plt.plot(model.dataset[compare_idx, :, i].squeeze().T, c='k', alpha=0.1)
        plt.plot(samples[:, :, i].squeeze().T, c='r', alpha=0.3)
        plt.show()

