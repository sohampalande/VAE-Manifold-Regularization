import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import MinMaxScaler

class TimeVAE(torch.nn.Module):

  def __init__(self, seq_len, feat_dim, latent_dim, hidden_layer_sizes, reconstruction_wt = 3.0, kernel_size=3, **kwargs):
    """Instantiate a timevae model.

    Args:
    - seq_len: the length of the window of the time series input
    - feat_dim: the dimensionality of each point in the time series (1 for univariate time series)
    - latent_dim: the dimensionality of the bottleneck layer of the TimeVAE model
    - hidden_layer_sizes: a list containing defining the number of neurons and hidden layers in the encoder and deocoder (ex: [100, 250])
    - reconstuction_wt (optional): the weight attached to the reconstruction error term of the loss function  
    - kernel_size (optional): the size of the kernel for the convolutional layers 
    """
      
    super(TimeVAE, self).__init__(**kwargs)

    self.seq_len = seq_len
    self.feat_dim = feat_dim
    self.latent_dim = latent_dim
    self.reconstruction_wt = reconstruction_wt
    self.hidden_layer_sizes = hidden_layer_sizes
  
    
    #intialize variables to track loss metrics
    self.total_loss_tracker = 0 
    self.reconstruction_loss_tracker = 0
    self.kl_loss_tracker = 0
    self.validation_loss_tracker = 0


    #initialize scaler for input/output data
    self.scaler = MinMaxScaler()

    #define encoder architecture
    modules = []  #create a list to hold the number of conv1d layers 
    
    feat_dim_temp = self.feat_dim
    for num_filters in self.hidden_layer_sizes:
      modules.append(torch.nn.Conv1d(in_channels=feat_dim_temp, out_channels=num_filters, kernel_size=kernel_size, stride=1, padding='same')) 
      modules.append(torch.nn.ReLU())
      feat_dim_temp = num_filters

    self.conv1d_encoder = torch.nn.Sequential(*modules)  #convert list of conv1d layers to nn.ModuleList
    self.flatten_layer = torch.nn.Flatten()

    #example to determine in_features for the linear layer
    x = torch.randn(1, self.feat_dim, self.seq_len)
    ex = self.conv1d_encoder(x)
    num_features = np.prod(ex.shape, 0)
    dim_conv_enc = ex.shape[1]
    #print(num_features)
    ####

    #get mean and variance
    self.mean = torch.nn.Linear(in_features=num_features, out_features=self.latent_dim)  
    self.log_var = torch.nn.Linear(in_features=num_features, out_features=self.latent_dim)


    #define decoder architecture
    self.linear_decoder = torch.nn.Linear(in_features=self.latent_dim, out_features=num_features)
    
    modules = []   #create a list to hold the number of conv1d transpose layers in the decoder
    feat_dim_temp = dim_conv_enc

    #reshape layer#

    for num_filters in list(reversed(self.hidden_layer_sizes))[:-1]:
      modules.append(torch.nn.ConvTranspose1d(in_channels=feat_dim_temp, out_channels=num_filters, kernel_size=kernel_size, stride=1, padding=1)) 
      modules.append(torch.nn.ReLU())
      feat_dim_temp = num_filters

    modules.append(torch.nn.ConvTranspose1d(in_channels=feat_dim_temp, out_channels=self.feat_dim, kernel_size=kernel_size, stride=1, padding=1))
    modules.append(torch.nn.ReLU())
    modules.append(torch.nn.Flatten())

    self.conv1d_tranpose_decoder = torch.nn.Sequential(*modules) #convert list of conv1d layers to nn.ModuleList

    #example to determine size of linear layer
    x = torch.randn((1, self.latent_dim))
    x = self.linear_decoder(x)

    x = torch.reshape(x, shape=(-1, self.hidden_layer_sizes[-1], self.seq_len))
    
    x = self.conv1d_tranpose_decoder(x)
    num_features = np.prod(x.size(), 0)
    ####

    self.decoder_output = torch.nn.Sequential(*[torch.nn.Linear(in_features=num_features, out_features=self.feat_dim*self.seq_len),
                                                torch.nn.Sigmoid()])

    #reshape layer#



  def encoder(self, x):
    #RETURNS: the result of one forward pass through the encoder
    #INPUTS:
    #x: data to be passed through encoder

    x = self.conv1d_encoder(x)
    
    x = self.flatten_layer(x)

    z_mean = self.mean(x)       #get mean and variance
    z_log_var = self.log_var(x)

    #get sampling
    batch = z_mean.size()[0]
    dim = z_mean.size()[1]

    epsilon = torch.randn(batch, dim) #generates batch x dim size tensor with values drawn from std. normal dist.
    encoder_output = z_mean + torch.exp(0.5*z_log_var)*epsilon

    return encoder_output, z_log_var, z_mean



  def decoder(self, x):
    #RETURNS: the result of one forward pass through the decoder
    #INPUTS:
    #x: data to be passed through encoder
    
    x = self.linear_decoder(x)
    
    x = torch.reshape(x, shape=(-1, self.hidden_layer_sizes[-1], self.seq_len))
    
    x = self.conv1d_tranpose_decoder(x)

    x = self.decoder_output(x)

    decoder_output = torch.reshape(x, shape=(-1, self.feat_dim, self.seq_len))

    return decoder_output



  def forward(self, x):
    #computes one forward propagation step through the defined network

    #forward pass through encoder
    encoder_output, z_log_var, z_mean = self.encoder(x)

    #forward pass through decoder
    decoder_output = self.decoder(encoder_output)

    return encoder_output, decoder_output, z_log_var, z_mean



  def get_reconstruction_loss(self, x, x_hat):
    #RETURNS: the reconstruction loss between the input x and the recontructed input x_hat
    #INPUT: 
    #x: the data input into the model
    #x_hat: the recontruction of input x

    def get_reconstruction_loss_by_axis(x, x_hat_c, axis):
      x_r = torch.mean(x, dim=axis)
      x_hat_c_r = torch.mean(x_hat_c, dim=axis)
      err = torch.square(torch.subtract(x_r, x_hat_c_r))
      loss = torch.sum(err)

      return loss
  
    err_time = torch.square(torch.subtract(x, x_hat))
    reconst_loss = torch.sum(err_time)

    # err_real = torch.square(torch.subtract(torch.fft.fft(x).real, torch.fft.fft(x_hat).real))
    # err_imag = torch.square(torch.subtract(torch.fft.fft(x).imag, torch.fft.fft(x_hat).imag))
    # err = err_real + err_imag + err_time
    # reconst_loss_1 = torch.sum(err)

    reconst_loss = reconst_loss  + 0*get_reconstruction_loss_by_axis(x, x_hat, axis=[1])

    return reconst_loss



  def get_prior_samples(self, num_samples):
    #RETURNS: num_samples number of random samples generated from random inputs (from Normal dist.) to the model's decoder
    #INPUT:
    #num_samples: the number of samples to be generated

    Z = torch.randn(num_samples, self.latent_dim)
    samples = self.decoder(Z)
    samples = torch.reshape(samples, shape=(-1, self.feat_dim, self.seq_len)).detach().numpy()
        
    #samples = samples.squeeze().T

    #inverse transform
    #samples = self.scaler.inverse_transform(samples)
    
    return samples



  def fit(self, dataset, dataset_name, seed, batch_size, lr=0.0010, val_set=0.1, file_path="", epochs=20):
    #RETURNS: the model fit to the data after training for specified number of epochs
    #INPUT: 
    #data: the dataset on which the model will be trained
    #seed: INT to set set seed for reproducible results
    #batch_size = set batch size for training
    #path: the path to the directory in which to save the model after training
    #epochs: the number of epochs to train the model 
    
    torch.random.manual_seed(seed) #set seed

    self.scaler = MinMaxScaler()

    scaled_data = self.scaler.fit_transform(dataset)
    scaled_data = torch.Tensor(scaled_data)
    
    #save scaler
    path = "./datasets/scaler_" + dataset_name + ".pkl"
    self.scaler.save_scaler(path)

    #split to get a validation set
    train_size = int((1-val_set) * len(scaled_data))
    validation_size = len(scaled_data) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(scaled_data, [train_size, validation_size])

    #convert dataset to tensor
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True) #create dataloader object to load data
    val_data = torch.utils.data.DataLoader(validation_dataset, batch_size, shuffle=True) 
  
    opt = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-07) #use Adam optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.1, patience=5)

    #store average losses after each epoch to plot loss curves
    kl_losses_all = []
    rec_losses_all = []
    total_losses_all = []
    validation_losses_all = []

    for epoch in range(epochs):

      total_losses = []
      kl_losses = []
      reconstruction_losses = []
      validation_losses = []
      
      for x in train_data:
        opt.zero_grad() ##sets the gradient for each param to 0 

        encoder_output, x_hat, z_log_var, z_mean = self(x)   #remove other return values if not used

        #compute reconstruction loss
        loss = self.get_reconstruction_loss(x, x_hat)

        #compute KL loss
        kl_loss = -0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
        kl_loss = torch.sum(torch.sum(kl_loss, dim=1))
        
        #compute total loss
        total_loss = self.reconstruction_wt * loss + kl_loss

        total_loss.backward()

        opt.step()

        #update metrics
        total_losses.append(total_loss.detach().numpy())
        kl_losses.append(kl_loss.detach().numpy())
        reconstruction_losses.append(loss.detach().numpy())

      for val_x in val_data:
        #compute validation loss
        val_encoder_output, val_x_hat, val_z_log_var, val_z_mean = self(val_x)

        #compute reconstruction loss
        val_loss = self.get_reconstruction_loss(val_x, val_x_hat)

        #compute KL loss
        val_kl_loss = -0.5 * (1 + val_z_log_var - torch.square(val_z_mean) - torch.exp(val_z_log_var))
        val_kl_loss = torch.sum(torch.sum(val_kl_loss, dim=1))
      
        #compute total loss
        val_total_loss = self.reconstruction_wt * val_loss + val_kl_loss

        #update scheduler
        #scheduler.step(val_total_loss)

        #record validation loss
        validation_losses.append(val_total_loss.detach().numpy())

      self.total_loss_tracker = np.mean(total_losses)
      self.kl_loss_tracker = np.mean(kl_losses)
      self.reconstruction_loss_tracker = np.mean(reconstruction_losses)
      self.validation_loss_tracker = np.mean(validation_losses)

      #record loss in list
      kl_losses_all.append(self.kl_loss_tracker)
      rec_losses_all.append(self.reconstruction_loss_tracker)
      total_losses_all.append(self.total_loss_tracker)
      validation_losses_all.append(self.validation_loss_tracker)

      print("Epoch: ", epoch)
      print(f'Total loss = {self.total_loss_tracker}')
      print(f'Reconstruction loss = {self.reconstruction_loss_tracker}')
      print(f'KL loss = {self.kl_loss_tracker}')
      print(f'Validation loss = {self.validation_loss_tracker}')
      print("")

    #plot total losses
    plt.plot(total_losses_all)
    plt.plot(kl_losses_all)
    plt.plot(rec_losses_all)
    plt.plot(validation_losses_all)
    plt.title('Model Loss', fontsize = 14)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Total Loss', 'KL Loss', 'Reconstruction Loss', 'Validation Loss'], loc='upper left')
    #plt.show()

    #save model
    if file_path!="":
      model_path = file_path + "/model.pt"
      torch.save(self, model_path)
      print("Model Saved!")

      path = file_path + "/model_training.png"
      plt.savefig(path)

    return self


