import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.tsvae_conv import ConvTimeSeriesVAE

# Global parameters
latent_dimension = 8
hidden_layers = [50, 100, 200]
reconstruction_wt = 1
kernel_size = 3
epochs = 2
batch_size = 32
lr = 0.001
seq_len = 30
features = ['Open', 'High', 'Low', 'Close', 'Adj Close']
time_name = ['Date']
dataset_name = 'AMZN_10-20'


if __name__ == '__main__':
    # Load dataset
    file_path = './datasets/' + dataset_name + '.csv'
    dataset = pd.read_csv(file_path)

    # Instantiate model and train
    model = ConvTimeSeriesVAE(seq_len=seq_len, dataset=dataset, time_column=time_name, feature_names=features,
                              latent_dim=latent_dimension, hidden_layer_sizes=hidden_layers,
                              reconstruction_wt=reconstruction_wt, kernel_size=kernel_size)

    model.fit(batch_size=batch_size, lr=lr, epochs=epochs)

    print("Model Trained.")

    # Output synthetic data samples as csv
    model.sample(100).to_csv('./datasets/sample_synthetic_data.csv')

    # Test to see quality of generated synthetic data
    N = 10
    samples = model.sample(N, return_dataframe=False)
    compare_idx = np.random.choice(model.dataset.shape[0], N, replace=False)
    for i in range(samples.shape[1]):
        plt.figure()
        plt.plot(samples[:, i, :].squeeze().T, c='r', alpha=0.3)
        plt.plot(model.dataset[compare_idx, i, :].squeeze().T, c='k', alpha=0.1)
        plt.show()


