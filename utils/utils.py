from random import sample
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.manifold import TSNE
import timevae
import pickle
import itertools
import seaborn as sns

sns.set_theme()

class MinMaxScaler():
    """Min Max normalizer.
    Args:
    - data: original data
    Returns:
    - norm_data: normalized data
    """
    def fit_transform(self, data): 
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data


    def fit(self, data):    
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini
        return self
      

    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-7)
        return scaled_data


    def inverse_transform(self, data):
        data *= self.range
        data += self.mini
        return data
    
      
    def save_scaler(self, filename):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)


class Visualize():

    def draw_orig_and_post_pred_sample_timeseries(orig, reconst, n, path=""):
        #RETURNS: plots of n randomly chosen samples from the input data and the corresponding reconstructions produced by the model
        #INPUT: 
        #orig: the training data on which the model is trained
        #reconst: the reconstructions of the the training data
        #n: the number of samples to be plotted
        fontsize = 14

        reconst = reconst.detach().numpy() 

        fig, axs = plt.subplots(n, 2, figsize=(10,6))
        i = 1
        for _ in range(n):
            rnd_idx = np.random.choice(len(orig))
        
            o = orig[rnd_idx]    #select 0th index if not using .squeeze()
            r = reconst[rnd_idx] 

            #swap axes to match tf plots
            o = np.swapaxes(o, 0, 1)
            r = np.swapaxes(r, 0, 1)

            plt.subplot(n, 2, i)
            plt.plot(o)
            plt.title("Original")
            i += 1

            plt.subplot(n, 2, i)
            plt.plot(r)
            plt.title("Sampled")
            i += 1

        fig.suptitle("Original vs Reconstructed Data", fontsize = fontsize)
        fig.tight_layout()

        if path!="":
            path = path + "/original_vs_reconstructions.png"
            plt.savefig(path)
        #plt.show()
    

    def draw_orig_and_post_pred_sample(orig, reconst, n, path=""):

        fontsize = 14
        reconst = reconst.detach().numpy() 

        fig, axs = plt.subplots(n, 2, figsize=(10,6))
        i = 1
        for _ in range(n):
            rnd_idx = np.random.choice(len(orig))
            o = orig[rnd_idx]
            r = reconst[rnd_idx]

            #swap axes to match TF plots
            o = np.swapaxes(o, 0, 1)
            r = np.swapaxes(r, 0, 1)

            plt.subplot(n, 2, i)
            plt.imshow(o, 
                # cmap='gray', 
                aspect='auto')
            plt.title("Original")
            i += 1

            plt.subplot(n, 2, i)
            plt.imshow(r, 
                # cmap='gray', 
                aspect='auto')
            plt.title("Sampled")
            i += 1

        fig.suptitle("Original vs Reconstructed Data", fontsize = fontsize)
        fig.tight_layout()

        if path!="":
            path = path + "/original_vs_reconstructions.png"
            plt.savefig(path)
        
        #plt.show()
    

    def plot_samples(model, n, inverse_transform=False, path=""):  
        #plots randomly generated samples that have been inversely transformed
        fontsize = 14
        plt.figure()
        samples = model.get_prior_samples(n)

        if inverse_transform:
            samples = model.scaler.inverse_transform(samples)

        plt.title('Generated Samples', fontsize=fontsize)

        plt.plot(samples[:, 0].T)

        if path!="":
            path = path + "/generated_samples.png"
            plt.savefig(path)

        #plt.show()


    def get_TSNE(dataset, samples, means=False, path=""):
        
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


    def get_TSNE_transfer(dataset1, dataset2, samples, means=False,  path=""):

        fontsize = 14
        
        prep_data1 = dataset1.squeeze()

        prep_data2 = dataset2.squeeze()

        prep_data_hat = samples.squeeze() 

        analysis_sample_no = prep_data1.shape[0] 
        
        colors = ["red" for i in range(analysis_sample_no)] + ["blue" for i in range(analysis_sample_no)] + ["lime" for i in range(analysis_sample_no*2)]

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data1, prep_data2), axis=0)
        prep_data_final = np.concatenate((prep_data_final, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)
        
        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(
            tsne_results[:analysis_sample_no, 0],
            tsne_results[:analysis_sample_no, 1],
            c=colors[:analysis_sample_no],
            alpha=0.95,
            label="Stock 1",
        )

        plt.scatter(
            tsne_results[analysis_sample_no : (len(colors)-(2*analysis_sample_no)), 0],
            tsne_results[analysis_sample_no : (len(colors)-(2*analysis_sample_no)), 1],
            c=colors[analysis_sample_no : (len(colors)-(2*analysis_sample_no))],
            alpha=0.17,
            label="Stock 2",
        )


        plt.scatter(
            tsne_results[(len(colors)-(2*analysis_sample_no)):, 0],
            tsne_results[(len(colors)-(2*analysis_sample_no)):, 1],
            c=colors[(len(colors)-(2*analysis_sample_no)):],
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
    
    
    def plot_slices(embeddings_real, embeddings_fake, path="", num_slices=4):

        fontsize = 14
        
        analysis_sample_no = embeddings_real.shape[0]
        colors = colors = ["red" for i in range(analysis_sample_no)] + ["blue" for i in range(analysis_sample_no)]

        # Get dimension pairs to plot
        dim = embeddings_real.shape[1]
        combo_set = list(itertools.combinations(list(range(dim)), 2))[:num_slices]

        fig = plt.figure()    
        fig.suptitle('t-SNE Plots of 2-Dimensional Subspaces of the Latent Space', fontsize = fontsize)
        
        for i in range(num_slices):
            plt.subplot(int(num_slices/2), 2, i+1)
            plt.scatter(    
                embeddings_real[:, combo_set[i][0]],
                embeddings_real[:, combo_set[i][1]],
                c=colors[:analysis_sample_no],
                alpha=0.2,
                label="Original",
            )

            plt.scatter(
                embeddings_fake[:, combo_set[i][0]],
                embeddings_fake[:, combo_set[i][1]],
                c=colors[analysis_sample_no:],
                alpha=0.2,
                label="Synthetic",
            )
        plt.legend()
        
        if path!="":
            path = path + "/latentspace_2Dslices.png"  #save plots
            plt.savefig(path)
