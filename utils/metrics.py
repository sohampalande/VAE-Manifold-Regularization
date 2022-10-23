import numpy as np
import torch
from torch import nn


def compute_f1(real_data, synthetic_data):
    """
    Compute the F1 score of a synthetic dataset with respect to a reference real dataset.

    Parameters
    ----------
    real_data: np.ndarray
        numpy array containing segmented time series from the "real" dataset. Shape of array should be (N, D, T),
        where N is the number of samples, D is the number of features (dimension), and T is the length of the segments
    synthetic_data: np.ndarray
        numpy array containing segmented time series from the synthetic dataset. Shape of array should be (N, D, T),
        where N is the number of samples, D is the number of features (dimension), and T is the length of the segments

    Returns
    -------
    f1_score
        harmonic mean of precision/recall scores (called f1 score), used for evaluating the fidelity of synthetic data
    """
    # Extract support of the time series for each time step
    data_real = [[real_data[k, :, u] for k in range(real_data.shape[0])] for u in range(real_data.shape[2])]
    real_support = [(np.min(data_real[v], axis=0), np.max(data_real[v], axis=0)) for v in range(real_data.shape[2])]

    data_synth = [[synthetic_data[k, :, u] for k in range(synthetic_data.shape[0])]
                  for u in range(synthetic_data.shape[2])]
    synth_support = [(np.min(data_synth[v], axis=0), np.max(data_synth[v], axis=0))
                     for v in range(synthetic_data.shape[2])]

    # Compute the average precision (coverage of synthetic data over real data support) across dimension
    # for each time step
    precision_t = [np.mean((data_synth[:][k] >= real_support[k][0]) * (data_synth[:][k] <= real_support[k][1]))
                   for k in range(real_data.shape[2])]
    # Obtain the mean precision taken across time steps
    precision = np.mean(precision_t)

    # Compute the average recall (coverage of real data over synthetic data support) across dimension
    # for each time step
    recall_t = [np.mean((data_real[:][k] >= synth_support[k][0]) * (data_real[:][k] <= synth_support[k][1]))
                for k in range(real_data.shape[2])]
    # Obtain the mean recall taken across time steps
    recall = np.mean(recall_t)

    # Take F1 score as the harmonic mean of the precision and recall
    f1_score = (2*precision*recall)/(precision+recall)

    return f1_score


def train_rnn(X, y, mode='LSTM', hidden_units=100, num_layers=2, num_epochs=10, batch_size=32, lr=1e-3):
    """
    Train a recurrent neural network model to forecast.

    Parameters
    ----------
    X: List[np.ndarray]
        training features used for 1-step ahead forecasting.
    y: List[np.ndarray]
        training labels to compare forecasted values to
    mode: str
        type of RNN model - should be from the list ['RNN', 'GRU', 'LSTM']
    hidden_units: int
        number of hidden units used in each layer of the RNN
    num_layers: int
        number of layers in the rnn
    num_epochs: int
        number of epochs to train the model for
    batch_size: int
        batch size used in training
    lr: float
        learning rate used for the optimizer

    Returns
    -------
    rnn_layer
        trained rnn network used to extract features from the time series seqments
    fc_layer
        trained fully connected that can be used to obtain the forecast from RNN features
    """
    # Obtain step size and input size
    steps, input_size = X[0].shape[0], X[0].shape[1]
    # Dictionary for choosing the model type
    rnn_dict = {"RNN": nn.RNN(input_size=input_size, hidden_size=hidden_units,
                              num_layers=num_layers, batch_first=True),
                "LSTM": nn.LSTM(input_size=input_size, hidden_size=hidden_units,
                                num_layers=num_layers, batch_first=True),
                "GRU": nn.GRU(input_size=input_size, hidden_size=hidden_units, num_layers=num_layers,
                              batch_first=True)}
    if mode not in rnn_dict.keys():
        raise ValueError("Mode specified does not correspond to valid RNN type")

    # Create data loader with train and test examples
    dataset = [(torch.Tensor(X[k]), torch.Tensor(y[k])) for k in range(len(X))]
    train_data = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    # Define the model
    rnn_layer = rnn_dict[mode]
    fc_layer = nn.Sequential(*[torch.nn.Flatten(),
                               torch.nn.Linear(in_features=hidden_units*steps, out_features=input_size)])

    # Fit the model
    opt = torch.optim.Adam(list(rnn_layer.parameters())+list(fc_layer.parameters()), lr=lr)  # Define optimizer
    for epoch in range(num_epochs):
        for x, y in train_data:
            # Zero out the gradient
            opt.zero_grad()
            # Forward pass through the model
            output, _ = rnn_layer(x)
            y_pred = fc_layer(output)
            # Compute the loss
            loss = torch.mean((y-y_pred)**2)

            # Back propagation
            loss.backward()
            opt.step()

    return rnn_layer, fc_layer


def compute_mae(real_data, synthetic_data):
    """
    Compute the MAE score in a "train-on-synthetic, test-on-real" experiment

    Parameters
    ----------
    real_data: np.ndarray
        numpy array containing segmented time series from the "real" dataset. Shape of array should be (N, D, T),
        where N is the number of samples, D is the number of features (dimension), and T is the length of the segments
    synthetic_data: np.ndarray
        numpy array containing segmented time series from the synthetic dataset. Shape of array should be (N, D, T),
        where N is the number of samples, D is the number of features (dimension), and T is the length of the segments

    Returns
    -------
    mae_score
        mean absolute error in the predictions of an RNN network trained for one-step ahead forecasting,
        when trained on the synthetic data and tested on the real data (TSTR)
    """
    # Get features and labels for real and synthetic data
    X_real = torch.Tensor(np.array([real_data[k, :, :-1].T for k in range(real_data.shape[0])]))
    Y_real = [real_data[k, :, -1].squeeze() for k in range(real_data.shape[0])]

    X_synth = [synthetic_data[k, :, :-1].T for k in range(synthetic_data.shape[0])]
    Y_synth = [synthetic_data[k, :, -1].squeeze() for k in range(synthetic_data.shape[0])]

    # Train the LSTM network
    batch_size = min([int(len(X_synth)/8), 128])
    hidden_units = int(len(X_synth)/(10*(X_synth[0].shape[1])))
    rnn, fc = train_rnn(X_synth, Y_synth, hidden_units=hidden_units, batch_size=batch_size)

    # Obtain MAE score by forward passing through the trained model
    with torch.no_grad():
        X_features, _ = rnn(X_real)
        Y_pred = fc(X_features).detach().numpy()
    MAE_score = np.mean(np.abs(np.array(Y_pred) - np.array(Y_real)))

    return MAE_score
