import numpy as np
import pandas as pd
import torch
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

def preprocess(loc, window_size, split, train_ratio=0.8):
  #RETURNS: the processed stock price dataset with each entry of length window_size
  #INPUT:
  #loc: the path to the directory containing the dataset and where the processed dataset will be saved
  #window_size: the size (period) of each time series entry in the dataset
  #split: 'True' or 'False'. Indicates whether to split the dataset into train and test sets
  #train_ratio: if split='True', indicates the ratio of the size of the train set to the test set. Default val 0.8  

  dataset = pd.read_csv(loc) #load data

  dataset = dataset["Adj Close"]  #select column Adj Close

  dataset = dataset.to_numpy()  #convert pandas series to numpy array

  print("The length of the dataset is:" , len(dataset))

  preprocessed_dataset = []
  
  for i in range((len(dataset)-window_size)+1):
    window = dataset[i:i+window_size]
    preprocessed_dataset.append(window)

  preprocessed_dataset = np.array(preprocessed_dataset) 
  print(preprocessed_dataset)
  print("The dimensions of the preprocessed dataset are: ", preprocessed_dataset.shape)

  # if split is true, split data intro train and test sets
  if split:
    split_index = int(train_ratio*len(preprocessed_dataset)) #calculate index of training set
    
    train_set = preprocessed_dataset[0:split_index]
    test_set = preprocessed_dataset[split_index:] 

    #save datasets

    file = loc[0:len(loc)-4]
    train_file = file + "_train"
    test_file = file + "_test"

    #reshape (if feat_dim not specified)
    
    if len(np.shape(train_set))<3:
    
      train_set = np.reshape(train_set, (train_set.shape[0], 1, train_set.shape[1]))
      test_set = np.reshape(test_set, (test_set.shape[0], 1, train_set.shape[1]))

    np.save(train_file, train_set)
    np.save(test_file, test_set)

    return
  
  else:

    #reshape
    if len(np.shape(preprocessed_dataset))<3:
      preprocessed_dataset = np.reshape(preprocessed_dataset, (preprocessed_dataset.shape[0], 1, preprocessed_dataset.shape[1]))

    #save dataset
    file = loc[0:len(loc)-4]
    file = file + "_preprocessed"
    np.save(file, preprocessed_dataset)

    return 

    
def preprocess_returns(loc, window_size, split, train_ratio=0.8):
  #RETURNS: the processed stock price dataset with each entry of length window_size
  #INPUT:
  #loc: the path to the directory containing the dataset and where the processed dataset will be saved
  #window_size: the size (period) of each time series entry in the dataset
  #split: 'True' or 'False'. Indicates whether to split the dataset into train and test sets
  #train_ratio: if split='True', indicates the ratio of the size of the train set to the test set. Default val 0.8  

  dataset = pd.read_csv(loc) #load data

  dataset = dataset["Adj Close"]  #select column Adj Close

  dataset = dataset.to_numpy()  #convert pandas series to numpy array

  print("The length of the dataset is:" , len(dataset))

  #calculate log_returns
  dataset = np.log(dataset[1:]) - np.log(dataset[:-1])

  preprocessed_dataset = []
  
  for i in range((len(dataset)-window_size)+1):
    window = dataset[i:i+window_size]
    preprocessed_dataset.append(window)

  preprocessed_dataset = np.array(preprocessed_dataset) 
  print(preprocessed_dataset)
  print("The dimensions of the preprocessed dataset are: ", preprocessed_dataset.shape)

  # if split is true, split data intro train and test sets
  if split:
    split_index = int(train_ratio*len(preprocessed_dataset)) #calculate index of training set
    
    train_set = preprocessed_dataset[0:split_index]
    test_set = preprocessed_dataset[split_index:] 

    #save datasets

    file = loc[0:len(loc)-4]
    train_file = file + "_train"
    test_file = file + "_test"

    #reshape (if feat_dim not specified)
    
    if len(np.shape(train_set))<3:
    
      train_set = np.reshape(train_set, (train_set.shape[0], 1, train_set.shape[1]))
      test_set = np.reshape(test_set, (test_set.shape[0], 1, train_set.shape[1]))

    np.save(train_file, train_set)
    np.save(test_file, test_set)

    return
  
  else:

    #reshape
    if len(np.shape(preprocessed_dataset))<3:
      preprocessed_dataset = np.reshape(preprocessed_dataset, (preprocessed_dataset.shape[0], 1, preprocessed_dataset.shape[1]))

    #save dataset
    file = loc[0:len(loc)-4]
    file = file + "_returns_preprocessed"
    np.save(file, preprocessed_dataset)

    return 


def combine_datasets(dataset1_path, dataset2_path, save_loc):

  dataset1 = np.load(dataset1_path)
  dataset2 = np.load(dataset2_path)
  print("Dataset 1 Shape: ", dataset1.shape)
  print("Dataset 2 Shape: ", dataset2.shape)

  print()

  dataset3 = np.append(dataset1, dataset2, axis=0)
  print("Dataset3 Shape: ", dataset3.shape)

  np.save(save_loc, dataset3)
  print("Dataset Saved!")


if __name__ == '__main__':
    
  print("haaaaa")

  #combine_datasets('./datasets/AMZN_10-20_preprocessed.npy', './datasets/GME_10-20_preprocessed.npy', './datasets/AMZN_GME_preprocessed.npy')

  ########### Generate Synthetic Data ####################
  num_samples = 2489

  # model_amzn = torch.load('./models/AMZN/seed_1/model.pt')
  # model_gme = torch.load('./models/GME/seed_1/model.pt')

  # samples_amzn = model_amzn.get_prior_samples(num_samples)
  # samples_gme = model_gme.get_prior_samples(num_samples)

  # print(samples_amzn.shape)
  # print(samples_gme.shape)

