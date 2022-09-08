import numpy as np
import matplotlib.pyplot as plt
from tsvae import TimeSeriesVAE

# Global parameters
latent_dimension = 8
hidden_layers = [16, 32, 64]
reconstruction_wt = 3
kernel_size = 3
epochs = 25
batch_size = 32
lr = 0.001
dataset_name = 'AMZN_10-20'


if __name__ == '__main__':
    # Load dataset
    file_path = './datasets/' + dataset_name + '_preprocessed.npy'
    dataset = np.load(file_path)
    seq_len = dataset.shape[-1]
    feat_dim = dataset.shape[1]

    # Instantiate model and train
    model = TimeSeriesVAE(seq_len=seq_len, feat_dim=feat_dim, latent_dim=latent_dimension,
                          hidden_layer_sizes=hidden_layers,reconstruction_wt=reconstruction_wt, kernel_size=kernel_size)

    model.fit(dataset=dataset, batch_size=batch_size, lr=lr, epochs=epochs)

    print("Model Trained.")

    plt.plot(model.sample(100).squeeze().T)
    plt.show()


