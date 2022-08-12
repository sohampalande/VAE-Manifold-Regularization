import torch
import numpy as np
import matplotlib.pyplot as plt; 
#plt.rcParams['figure.dpi'] = 150
import seaborn as sns
from sklearn.manifold import TSNE
import cv2

sns.set_theme()

params = {'font.size': 18.0,
 'axes.labelsize': 'small',
 'axes.titlesize': 'medium',
 'xtick.labelsize': 'small',
 'ytick.labelsize': 'small',
 'legend.fontsize': 'small',
 'axes.linewidth': 0.8,
 'grid.linewidth': 0.8,
 'lines.linewidth': 1.5,
 'lines.markersize': 6.0,
 'patch.linewidth': 1.0,
 'xtick.major.width': 0.8,
 'ytick.major.width': 0.8,
 'xtick.minor.width': 0.6,
 'ytick.minor.width': 0.6,
 'xtick.major.size': 3.5,
 'ytick.major.size': 3.5,
 'xtick.minor.size': 2.0,
 'ytick.minor.size': 2.0,
 'legend.title_fontsize': 12}

sns.set_context(params)



class Visualize():

    def create_video(image_filepath, video_filepath, fps=20):
        """Outputs a video from a sequence of images in the .avi format to the specified location.

        Args:
        - image_filepath (string): path to directory containing the sequence of images 
        - video_filepath (string): target location to save output video file
        - fps (int): (Frames per Second) specifies the frames per second of the video file (15-24 works best) 
        """
        
        vid_capture = cv2.VideoCapture(image_filepath)

        if (vid_capture.isOpened() == False):
            print("Error opening the video file")
        
        else:
            # Get frame rate information
            fps = int(vid_capture.get(5))
            print("Frame Rate : ",fps,"frames per second") 

            # Get frame count
            frame_count = vid_capture.get(7)
            print("Frame count : ", frame_count)


        #write video
        frame_width = int(vid_capture.get(3))
        frame_height = int(vid_capture.get(4))
        frame_size = (frame_width,frame_height)

        # Initialize video writer object
        output = cv2.VideoWriter(video_filepath, cv2.VideoWriter_fourcc('M','J','P','G'), fps, frame_size)
        
        while(vid_capture.isOpened()):

            # vid_capture.read() methods returns a tuple, first element is a bool
            # and the second is frame
            ret, frame = vid_capture.read()
            
            if ret == True:
                # Write the frame to the output files
                output.write(frame)

            else:
                print("Stream disconnected")
                break

        # Release the objects
        vid_capture.release()
        output.release()

        print("Video Saved!")

        return


    def imshow(image, ax=None, title=None, normalize=True):
        """Plot an image from a DataLoader.

        Args:
        - image: a Torch tensor with shape- [sample_num, height, width]
        """

        image = image.detach().numpy()
        image = np.squeeze(image)
        if ax is None:
            fig, ax = plt.subplots()
        #image = image.numpy().transpose((1, 2, 0))

        if normalize:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)

        ax.imshow(image)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', length=0)
        ax.set_xticklabels('')
        ax.set_yticklabels('')

        return ax


    #randomly choose a sample 
    # sample from the normal distribution (that is the assumption)

    def generate_samples(model, n):
        """Generate and plot synthetic from given model .

        Args:
        - model: a pytorch model from which to generate synthetic data
        - n:  number of synthetic data samples to generate from model
        """

        samples = []

        for i in range(n):
            z = torch.randn((1, model.latent_dim))
            x_hat = model.decoder(z)

            x_hat = x_hat.reshape(28, 28).detach().numpy()

            samples.append(x_hat)
        
        samples = np.array(samples)

        return samples


    #visualize the latent space by plotting the latent vectors produced by the autoencoder 
    #for a batch of inputs

    def plot_latent(vae_model, data, num_batches=100, path="", show_plots=True, lim=[-3, 3]):
        """Plot the latent space embeddings of a given VAE model's encoder.

        Args:
        - vae_model: a pytorch model 
        - data: the Dataloader corresponding to the data that the model was trained on
        - num_batches (int): number of batches to plot
        - path (string): specify the path to save the plot
        - show_plots (bool): True or False: if True, display plot; if false do not display
        - lim (list- 2 elements):  specify the limits of the x-axis and y-axis of plot
        """

        plt.figure()
        plt.ylim(lim[0],lim[1])
        plt.xlim(lim[0],lim[1])
        for i, (x,y) in enumerate(data):
            z, mu, sigma = vae_model.encoder(x)
            mu = mu.detach().numpy()
            plt.title('Latent Space of Train Data', fontsize=12)
            plt.scatter(mu[:, 0], mu[:, 1], c=y, alpha=0.3, cmap='tab10')

        plt.colorbar()
            
        
        if path!="":
            plt.savefig(path)
        
        if not show_plots:
            plt.close()
        
        return


    #visualize the latent space by plotting the latent vectors produced by the autoencoder 
    #for a batch of inputs

    def plot_latent_regularization(vae_model, data, num_batches=100, path="", show_plots=True, lim=[-3, 3]):
        """Plot the latent space embeddings of a given VAE model's encoder. For VAE with manifold regularization

        Args:
        - vae_model: a pytorch model 
        - data: the Dataloader corresponding to the data that the model was trained on
        - num_batches (int): number of batches to plot
        - path (string): specify the path to save the plot
        - show_plots (bool): True or False: if True, display plot; if false do not display
        - lim (list- 2 elements):  specify the limits of the x-axis and y-axis of plot
        """

        plt.figure()
        plt.ylim(lim[0],lim[1])
        plt.xlim(lim[0],lim[1])
        for i, (x,y,z) in enumerate(data):
            z, mu, sigma = vae_model.encoder(x)
            mu = mu.detach().numpy()
            plt.title('Latent Space of Train Data', fontsize=12)
            plt.scatter(mu[:, 0], mu[:, 1], c=y, alpha=0.3, cmap='tab10')

        plt.colorbar()
            
        
        if path!="":
            plt.savefig(path)
        
        if not show_plots:
            plt.close()
        
        return


    #randomly sample from the latent space and plot the reconstructed output from the decoder

    def plot_reconstructions_space(vae_model, r0=(-1, 1), r1=(-1,1), n=12, path="", show_plots=True):
        """Plot the image reconstructions of the latent space embeddings of a given VAE model's encoder.

        Args:
        - vae_model: a pytorch model 
        - r0: specify the interval from which to uniformly sample n points for the first coordinate
        - r1: specify the interval from which to uniformly sample n points for the second coordinate
        - n (int): generates n^2 recontructions uniformly sampled from the latent space using r0, r1
        - path (string): specify the path to save the plot
        - show_plots (bool): True or False: if True, display plot; if false do not display
        """

        w = 28
        
        img = np.zeros((n*w, n*w)) #we will create a grid of 144 samples as a single image which we initlaize to 0

        plt.figure()

        #unifoormly sample images from the latent space 
        for i,y in enumerate(np.linspace(*r1, n)):  #np.linspace generates n evenly spaced numbers in the range given by r1 
            for j, x in enumerate(np.linspace(*r0, n)):

                z = torch.Tensor([[x, y]]) #create tensor of randomly sampled 2-d vector of the latent space
                x_hat = vae_model.decoder(z) #generate the reconstruction

                x_hat = x_hat.reshape(28,28).detach().numpy()
                img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
        
        plt.title('Latent Space Reconstructions')

        plt.imshow(img, extent=[*r0, *r1]) #extent sets the coordinates for the axes
        
        if path!="":
            plt.savefig(path)
        
        if not show_plots: 
            plt.close()
        
        return


    #randomly sample from the latent space and plot the reconstructed output from the decoder

    def plot_reconstructions_normal(vae_model, n=5, path="", show_plots=True):
        """Generate synthetic data.

        Args:
        - vae_model: a pytorch model from which to generate synthetic data
        - n (int): number of synthetic data points to generate
        - path (string): specify the path to save the plot
        - - show_plots (bool): True or False: if True, display plot; if false do not display
        """

        w = 28
        
        img = np.zeros((w, n*w)) #we will create a grid of 144 samples as a single image which we initlaize to 0
        
        plt.title('Synthetic Data')
        #unifoormly sample images from the latent space 
        for i in range(n):  #np.linspace generates n evenly spaced numbers in the range given by r1 

            random_sample = np.random.randn(vae_model.latent_dim)

            z = torch.Tensor([random_sample]) #create tensor of randomly sampled 2-d vector of the latent space
            x_hat = vae_model.decoder(z) #generate the reconstruction

            x_hat = x_hat.reshape(28,28).detach().numpy()
            img[:, w*i:w+(w*i)] = x_hat
        
        plt.imshow(img) #extent sets the coordinates for the axes

        if path!="":
            plt.savefig(path)

        if not show_plots:
            plt.close()
        
        return


    def get_TSNE(dataset, samples, means=False, path=""):
        """Plot t-SNE of original and synthetic data.

        Args:
        - dataset (numpy array): original dataset
        - samples (numpy array): synthetic dataset
        - means (bool): if true changes save directory of plot
        - path (string): specify the path to save the plot
        """
        
        fontsize = 14

        prep_data = dataset.squeeze()

        prep_data_hat = samples.squeeze() 

        analysis_sample_no = prep_data.shape[0]
        colors = ["red" for i in range(analysis_sample_no)] + ["blue" for i in range(analysis_sample_no)]

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)
        
        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(
            tsne_results[:analysis_sample_no, 0],
            tsne_results[:analysis_sample_no, 1],
            c=colors[:analysis_sample_no],
            alpha=0.3,
            label="Original",
        )

        plt.scatter(
            tsne_results[analysis_sample_no:, 0],
            tsne_results[analysis_sample_no:, 1],
            c=colors[analysis_sample_no:],
            alpha=0.17,
            label="Synthetic",
        )
        plt.legend()
        plt.title('t-SNE Plot', fontsize = fontsize)
        
        if path!="":
            if means:
                path = path + "/t-sne_plot_means.png"
            else:
                path = path + "/t-sne_plot.png"
            plt.savefig(path)
        
        #plt.show()

        return


    def get_TSNE_transfer(dataset1, dataset2, samples, means=False,  path=""):
        """Plot t-SNE of two training datasets and their correpsonding synthetic data.

        Args:
        - dataset1 (numpy array): original dataset1
        - dataset2 (numpy array): original dataset2
        - samples (numpy array): synthetic dataset
        - means (bool): if true changes save directory of plot
        - path (string): specify the path to save the plot
        """

        fontsize = 14
        
        prep_data1 = dataset1.squeeze()

        prep_data2 = dataset2.squeeze()

        prep_data_hat = samples.squeeze() 

        analysis_sample_no_1 = prep_data1.shape[0] 
        analysis_sample_no_2 = prep_data2.shape[0] 
        analysis_sample_no_3 = prep_data_hat.shape[0] 
        
        colors = ["red" for i in range(analysis_sample_no_1)] + ["blue" for i in range(analysis_sample_no_2)] + ["lime" for i in range(analysis_sample_no_3)]

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data1, prep_data2), axis=0)
        prep_data_final = np.concatenate((prep_data_final, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)
        
        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(
            tsne_results[:analysis_sample_no_1, 0],
            tsne_results[:analysis_sample_no_1, 1],
            c=colors[:analysis_sample_no_1],
            alpha=0.95,
            label="Class 1",
        )

        plt.scatter(
            tsne_results[analysis_sample_no_1 : analysis_sample_no_1+analysis_sample_no_2, 0],
            tsne_results[analysis_sample_no_1 : analysis_sample_no_1+analysis_sample_no_2, 1],
            c=colors[analysis_sample_no_1 : analysis_sample_no_1+analysis_sample_no_2],
            alpha=0.17,
            label="Class 2",
        )


        plt.scatter(
            tsne_results[analysis_sample_no_1+analysis_sample_no_2:, 0],
            tsne_results[analysis_sample_no_1+analysis_sample_no_2:, 1],
            c=colors[analysis_sample_no_1+analysis_sample_no_2:],
            alpha=0.07,
            label="Synthetic",
        )
        plt.legend()
        plt.title('t-SNE Plot', fontsize = fontsize)

        if path!="":
            if means:
                path = path + "/t-sne_plot_means.png"
            else:
                path = path + "/t-sne_plot.png"
            plt.savefig(path)

        #plt.show()

    #returns entire dataset from a given DataLoader
    def get_dataset_from_loader(dataset_loader):
        """Returns a numpy array from the Dataloader.

        Args:
        - dataset1 (numpy array): original dataset1
        """

        dataset = []

        for x,y in dataset_loader:
            
            x = np.array(x)
            x = np.squeeze(x)

            for img in x:
                dataset.append(img)

        dataset = np.array(dataset)

        return dataset


class MNISTSampler(torch.utils.data.sampler.Sampler):
    
    def __init__(self, mask, data_source):
        self.mask = mask
        self.data_source = data_source

    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(self.mask)])

    def __len__(self):
        return len(self.data_source)
    


class GraphConstruction():
    def get_embedding_matrix(model, samples=1000, gamma=1, num_neighbors=10):
        """Computes the adjacency matrix of a graph approximating the manifold corresponding to the latent space embeddings of
        synthetic data generated by the given model.

        Returns:
        - adjancency matrix and corresponding synthetic data

        Args:
        - mdoel: model to generate synthetic data 
        - samples (int): number of synthetic data points to use for graph construction
        - gamma (float): hyperparamter controlling ...
        - num_neighbors (int): used when using KNN
        """

        def get_symmteric_KL(mu, sigma, i, j):

            #convert to numpy
            mu = mu.detach().numpy()
            sigma = sigma.detach().numpy()

            mu_i = mu[i]
            mu_j = mu[j]

            d_z = mu_i.shape[0]

            sigma_i = sigma[i]
            sigma_j = sigma[j]

            #compute i|j
            term_1 = np.divide(np.power(sigma_i, 2), np.power(sigma_j, 2))
            term_2 = np.divide(np.power(np.subtract(mu_i, mu_j), 2), np.power(sigma_j, 2))
            term_3 = np.subtract(np.log(np.power(sigma_j, 2)), np.log(np.power(sigma_i, 2)))
            
            total_ij = 0.5*sum(term_1 + term_2 + term_3 ) - (0.5*d_z) 

            #compute j|i
            term_1 = np.divide(np.power(sigma_j, 2), np.power(sigma_i, 2))
            term_2 = np.divide(np.power(np.subtract(mu_j, mu_i), 2), np.power(sigma_i, 2))
            term_3 = np.subtract(np.log(np.power(sigma_i, 2)), np.log(np.power(sigma_j, 2)))

            total_ji = 0.5*sum(term_1 + term_2 + term_3) - (0.5*d_z) 

            return 0.5*(total_ij + total_ji) 

        #generate synthetic data from given model
        synthetic_data = Visualize.generate_samples(model=model, n=samples) #(num_samples, height, width)
        synthetic_data = np.reshape(synthetic_data, newshape=(synthetic_data.shape[0], 1, synthetic_data.shape[1], synthetic_data.shape[2])) #reshape to (num_samples, 1, 28, 28) 
        synthetic_data = torch.tensor(synthetic_data) #convert to tensor

        #forward pass through encoder to obtain Gaussian embeddings
        gaussian_embeddings, mu, sigma = model.encoder(synthetic_data)
        gaussian_embeddings = gaussian_embeddings.detach().numpy() #(num_samples, latent_dim=2)

        W = np.ones(shape = (gaussian_embeddings.shape[0], gaussian_embeddings.shape[0])) #initialize adjacency matrix

        #compute distance between each pair of embeddings
        for i in range(gaussian_embeddings.shape[0]):
            for j in range(gaussian_embeddings.shape[0]):

                #if distance already computed, skip
                if W[i,j] != 1:
                    continue

                if i == j:
                    W[i,j] = 0
                    continue
                
                #compute distance
                dist = get_symmteric_KL(mu=mu, sigma=sigma, i=i, j=j) 

                #update adjacency matrix
                W[i,j] = np.exp(-1 * gamma * dist**2)
                W[j,i] = W[i,j] #since the matrix is symmetric
        
        # Convert to KNN graph
        # A = np.zeros(W.shape)
        # for i in range(gaussian_embeddings.shape[0]):
        #   threshold = np.sort(W[i])[-num_neighbors]
        #   for j in range(gaussian_embeddings.shape[0]):
        #     if W[i, j] >= threshold:
        #       A[i, j] = 1
        #       A[j, i] = 1

        synthetic_data = synthetic_data.detach().numpy() #convert synthetic_data to numpy array
        
        return W, synthetic_data   

    
    def get_laplacian(W):
        """Returns the Laplacian corresponding to an adjacency matrix.

        Args:
        - W (numpy array): adjacency matrix
        """

        D = np.zeros(shape=(W.shape[0], W.shape[1]))

        for i in range(W.shape[0]):
            D[i,i] = sum(W[i])
        
        L = D - W

        return L


    def check_symmetric(a, rtol=1e-05, atol=1e-08):
        """Checks whether a given adjacency matrix is symmetric.
        
        Returns:
        - (bool): True/False

        Args:
        - a (numpy array): adjacency matrix
        """
        return np.allclose(a, a.T, rtol=rtol, atol=atol)


    

    
