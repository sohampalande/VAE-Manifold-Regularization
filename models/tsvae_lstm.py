import pandas as pd
import torch
from typing import List
import numpy as np
from utils.utils import MinMaxScaler
import torch.nn.functional as F


class LSTMTimeSeriesVAE(torch.nn.Module):
    def __init__(self, dataset, time_column, feature_names, latent_dim, hidden_size, num_layers=2,
                 reconstruction_wt=3.0, segment_length=30, segment_stride=1, **kwargs):
        """
        Instantiate a VAE model for time series data (using LSTMs)

        Parameters
        ----------
        seq_len: int
            the length of the window of the time series input
        dataset:
            Pandas dataframe containing the time series dataset
        time_column: str
            column of the dataset that corresponds to the time component
        features: List[str]
            columns of the dataset that correspond the features
        latent_dim: int
            the dimensionality of the bottleneck layer of the TimeVAE model
        hidden_layer_sizes: List[int]
            a list containing defining the number of neurons and hidden layers in the encoder and
            decoder (ex: [100, 250])
        reconstruction_wt: float
            the weight attached to the reconstruction error term of the loss function
        kernel_size: int
            the size of the kernel for the convolutional layers
        """
        super(LSTMTimeSeriesVAE, self).__init__()

        # Set parameters as attributes of class
        self.raw_data = dataset
        self.time_index = time_column
        self.features = feature_names
        self.feat_dim = len(self.features)
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        self.seq_len = segment_length
        self.segment_length = segment_length
        self.segment_stride = segment_stride
        self.scaler = None

        # Preprocess the data to be used by the time series generation method
        self.dataset = self.prepare_dataset()

        # Initialize variables to track loss metrics
        self.total_loss_tracker = 0.0
        self.reconstruction_loss_tracker = 0.0
        self.kl_loss_tracker = 0.0
        self.validation_loss_tracker = 0.0

        # Initialize attributes related to architecture
        self.conv1d_encoder = None
        self.encoder_feature_size = None
        self.dim_conv = None
        self.mean = None
        self.log_var = None
        self.linear_decoder = None
        self.conv1d_tranpose_decoder = None
        self.decoder_output = None

        # Define the encoder and decoder networks
        # Encoder:
        self.lstm_encoder = torch.nn.LSTM(input_size=self.feat_dim, hidden_size=self.hidden_dim,
                                          num_layers=self.num_layers,  batch_first=True)

        # Get mean and variance of latent distribution
        self.mean = torch.nn.Linear(in_features=self.hidden_dim*self.seq_len, out_features=self.latent_dim)
        self.log_var = torch.nn.Linear(in_features=self.hidden_dim*self.seq_len, out_features=self.latent_dim)

        # Decoder:
        self.decoder_fc = torch.nn.Linear(in_features=self.latent_dim, out_features=self.hidden_dim*self.seq_len)
        self.lstm_decoder_1 = torch.nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim,
                                            num_layers=self.num_layers, batch_first=True)

        self.lstm_decoder_2 = torch.nn.LSTM(input_size=self.hidden_dim, hidden_size=self.feat_dim, num_layers=1)

        self.flatten_layer = torch.nn.Flatten()

    def prepare_dataset(self):
        # Sort the pandas dataframe by the time column (ascending) and convert to numpy array
        temp_data = self.raw_data.sort_values(by=self.time_index)[self.features].to_numpy()
        # Generate the segmented data
        segmented_dataset = []
        for i in range(temp_data.shape[0]-self.segment_length):
            segmented_dataset.append(np.array(temp_data[i:i+self.segment_length]))
        return np.array(segmented_dataset)

    def encoder(self, x):
        """
        Encoder network (forward pass). The encoder network is used to encode the time series into a Gaussian random
        vector.

        Parameters
        ----------
        x: tensor
            input time series data to be encoded

        Returns
        -------
        embeddings:
            samples from posterior distribution in latent space (sampled VAE embeddings)
        z_log_var:
            log variances of diagonal Gaussian posterior distribution of VAE latent variable
        z_mean:
            means of Gaussian posterior distribution of VAE latent variable
        """
        # Forward pass through convolutional layers
        x, _ = self.lstm_encoder(x)
        x = self.flatten_layer(x)

        # Obtain mean and variance
        z_mean = self.mean(x)  # get mean and variance
        z_log_var = self.log_var(x)

        # Sample from posterior distribution of latent variable
        batch = z_mean.size()[0]
        dim = z_mean.size()[1]

        # Obtain embeddings by applying transformation
        epsilon = torch.randn(batch, dim)  # generates batch x dim size tensor with values drawn from std. normal dist.
        embeddings = z_mean + torch.exp(0.5 * z_log_var) * epsilon

        return embeddings, z_log_var, z_mean

    def decoder(self, z):
        """
        Decoder network (forward pass). Sampled embeddings can be passed through the decoder network to reconstruct the
        time series for which that embedding encoded.

        Parameters
        ----------
        z:
            embedding to be decoded

        Returns
        -------
        samples
            samples from posterior distribution in latent space (sampled VAE embeddings)
        """
        # pad the embedding with zeros
        z = self.decoder_fc(z).view(-1, self.seq_len, self.hidden_dim)

        x, _ = self.lstm_decoder_1(z)
        x, _ = self.lstm_decoder_2(x)
        return x

    def forward(self, x):
        """
        End to end forward pass of a time series through the VAE model.

        Parameters
        ----------
        x:
            input time series

        Returns
        -------
        embeddings:
            samples from posterior distribution in latent space (sampled VAE embeddings)
        reconstructions:
            reconstructions of the input time series data
        z_log_var:
            log variances of diagonal Gaussian posterior distribution of VAE latent variable
        z_mean:
            means of Gaussian posterior distribution of VAE latent variable
        """
        # Forward pass through encoder
        embeddings, z_log_var, z_mean = self.encoder(x)

        # Forward pass through decoder
        reconstructions = self.decoder(embeddings)

        return embeddings, reconstructions, z_log_var, z_mean

    def get_reconstruction_loss(self, x, x_hat):
        """
        Compute the reconstruction loss (in terms of MSE) between two time series.

        Parameters
        ----------
        x:
            input time series
        x_hat:
            reconstructed time series

        Returns
        -------
        loss
            reconstruction loss
        """
        err_time = torch.square(torch.subtract(x, x_hat))
        loss = torch.sum(err_time)

        return loss

    def sample(self, num_samples, return_dataframe=True):
        """
        Method used at inference time to draw samples from the VAE model.

        Parameters
        ----------
        num_samples:
            number of time series to draw

        Returns
        -------
        samples:
            samples drawn from the model
        """

        Z = torch.randn(num_samples, self.latent_dim)
        samples = self.decoder(Z)
        samples = torch.reshape(samples, shape=(-1, self.seq_len, self.feat_dim)).detach().numpy()
        samples = self.scaler.inverse_transform(samples)

        # Return as dataframe in same format as raw data?
        if return_dataframe:
            synthetic_data = pd.DataFrame(columns=['seg_index', 'time_index']+self.features)
            for n in range(num_samples):
                sample_df = pd.DataFrame(samples[n].T, columns=self.features)
                sample_df['seg_index'] = (n*np.ones(self.segment_length)).astype(int)
                sample_df['time_index'] = (np.linspace(0, self.segment_length-1, self.segment_length)).astype(int)
                # Concat to output data
                synthetic_data = pd.concat([synthetic_data, sample_df])
        else:
            synthetic_data = samples

        return synthetic_data

    def fit(self, batch_size, lr=0.001, epochs=20, verbose=True):
        """
        Fit function used to train the VAE model given a time series dataset.

        Parameters
        ----------
        dataset:
            time series dataset as a numpy array with shape (N, D, T), where N is the total number of samples in the
            dataset, D is the dimensionality of each multivariate time series, and T is the length of each time series
        batch_size:
            batch size to be used to compute each stochastic gradient
        lr:
            learning rate of the optimizer
        epochs:
            maximum number of epochs to train the model for
        verbose: bool
            If true, print progress of training the model
        """
        self.scaler = MinMaxScaler(by_axis=0)
        scaled_data = self.scaler.fit_transform(self.dataset)
        scaled_data = torch.Tensor(scaled_data)

        # Create dataloader
        train_data = torch.utils.data.DataLoader(scaled_data, batch_size, shuffle=True)

        opt = torch.optim.Adam(self.parameters(), lr=lr)  # use Adam optimizer

        # store average losses after each epoch to plot loss curves
        kl_losses_all = []
        rec_losses_all = []
        total_losses_all = []

        for epoch in range(epochs):

            total_losses = []
            kl_losses = []
            reconstruction_losses = []

            for x in train_data:
                opt.zero_grad()  ##sets the gradient for each param to 0

                encoder_output, x_hat, z_log_var, z_mean = self(x)  # remove other return values if not used

                # compute reconstruction loss
                loss = self.get_reconstruction_loss(x, x_hat)

                # compute KL loss
                kl_loss = -0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
                kl_loss = torch.sum(torch.sum(kl_loss, dim=1))

                # compute total loss
                total_loss = self.reconstruction_wt * loss + kl_loss

                total_loss.backward()

                opt.step()

                # update metrics
                total_losses.append(total_loss.detach().numpy())
                kl_losses.append(kl_loss.detach().numpy())
                reconstruction_losses.append(loss.detach().numpy())

            self.total_loss_tracker = np.mean(total_losses)
            self.kl_loss_tracker = np.mean(kl_losses)
            self.reconstruction_loss_tracker = np.mean(reconstruction_losses)

            # record loss in list
            kl_losses_all.append(self.kl_loss_tracker)
            rec_losses_all.append(self.reconstruction_loss_tracker)
            total_losses_all.append(self.total_loss_tracker)

            if verbose:
                print(f'Epoch: {epoch}')
                print(f'Total loss = {self.total_loss_tracker: .4f}')
                print(f'Reconstruction loss = {self.reconstruction_loss_tracker: .4f}')
                print(f'KL loss = {self.kl_loss_tracker: .4f}')
                print("")
