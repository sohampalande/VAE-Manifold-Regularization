import numpy as np
import pandas as pd
import torch
import pytest
from models.tsvae_conv import ConvTimeSeriesVAE


def create_sample_model(time_length=1000, num_features=5, segment_length=50, latent_dim=8,
                        reconstruction_wt=3.0):
    # Create things that will fill in dataframe
    time_steps = np.array([np.linspace(0, 1, time_length)]).T
    features = np.random.normal(loc=0, scale=1, size=(time_length, num_features))
    # Create names of dataframe
    time_name = 'time'
    feature_names = [str(i) for i in range(num_features)]
    # Create pandas dataframe with the data
    dataset = pd.DataFrame(data=np.concatenate([time_steps, features], axis=1),
                           columns=[time_name]+feature_names)
    # Create a model to test on
    model = ConvTimeSeriesVAE(dataset=dataset, time_column=time_name, feature_names=feature_names,
                              segment_length=segment_length, reconstruction_wt=reconstruction_wt,
                              latent_dim=latent_dim)
    return model


def test_constructor():

    # Check if certain attributes give problems
    with pytest.raises(ValueError):
        model = create_sample_model(latent_dim=-1)
    with pytest.raises(ValueError):
        model = create_sample_model(reconstruction_wt=-17)
    with pytest.raises(ValueError):
        model = create_sample_model(time_length=1000, segment_length=1001)

    # Test dataset segmentation
    model = create_sample_model()
    assert isinstance(model.dataset, np.ndarray)
    assert model.dataset.shape[0] == model.raw_data.shape[0]-model.segment_length
    assert model.dataset.shape[1] == model.raw_data.shape[1]-1
    assert model.dataset.shape[2] == model.segment_length

    # Test minmax scaling is working
    assert isinstance(model.scaled_dataset, np.ndarray)
    assert np.min(model.scaled_dataset) >= 0
    assert np.max(model.scaled_dataset) <= 1

    # Test encoder definition
    assert isinstance(model.conv1d_encoder, torch.nn.Module)
    assert isinstance(model.mean, torch.nn.Module)
    assert isinstance(model.log_var, torch.nn.Module)

    # Test decoder definition
    assert isinstance(model.linear_decoder, torch.nn.Module)
    assert isinstance(model.conv1d_tranpose_decoder, torch.nn.Module)
    assert isinstance(model.decoder_output, torch.nn.Module)


def test_encoder_forward_pass():
    model = create_sample_model()

    # Test for single element in the training dataset and for batches
    single_ts = torch.Tensor(model.dataset[0]).view(-1, model.feat_dim, model.segment_length)
    batch_ts = torch.Tensor(model.dataset[:16]).view(-1, model.feat_dim, model.segment_length)
    examples = [single_ts, batch_ts]
    for example in examples:
        # Forward pass the example and verify it
        latent_code, log_var, mean = model.encoder(example)
        # Check types
        assert torch.is_tensor(latent_code)
        assert torch.is_tensor(log_var)
        assert torch.is_tensor(mean)
        # Verify shapes
        assert latent_code.shape[0] == example.shape[0]
        assert log_var.shape[0] == example.shape[0]
        assert mean.shape[0] == example.shape[0]
        assert latent_code.shape[1] == model.latent_dim
        assert log_var.shape[1] == model.latent_dim
        assert mean.shape[1] == model.latent_dim
        # Verify exp(log_var) is strictly positive
        assert torch.all((torch.exp(log_var) > 0).bool())


def test_decoder_forward_pass():
    model = create_sample_model()

    # Test for single element in the training dataset and for batches
    latent_code = torch.randn(size=(1, model.latent_dim))
    latent_code_batch = torch.randn(size=(10, model.latent_dim))
    examples = [latent_code, latent_code_batch]
    for example in examples:
        # Forward pass the example and verify it
        decoded_ts = model.decoder(example)
        # Check type
        assert torch.is_tensor(decoded_ts)
        # Verify shapes (generating time series with correct number of features and segment length)
        assert decoded_ts.shape[0] == example.shape[0]
        assert decoded_ts.shape[1] == model.feat_dim
        assert decoded_ts.shape[2] == model.segment_length


def test_sampling_method():
    model = create_sample_model()
    # Draw samples from the model (returning the numpy array) - verify returns batches
    max_batch_size = 16
    for n_samples in range(max_batch_size):
        synthetic_sample = model.sample(num_samples=n_samples, return_dataframe=False)
        assert isinstance(synthetic_sample, np.ndarray)
        assert synthetic_sample.shape[0] == n_samples
        assert synthetic_sample.shape[1] == model.feat_dim
        assert synthetic_sample.shape[2] == model.segment_length
        # Verify samples are contained in the range training data (minmax scaling inverse transform works)
        assert np.all(np.min(synthetic_sample, axis=2) >= model.scaler.mini.squeeze())
        assert np.all(np.max(synthetic_sample, axis=2) <= (model.scaler.mini.squeeze()
                                                           + model.scaler.range.squeeze()))

    # Draw samples from the model (returning the DataFrame) - verify returns batches
    for n_samples in range(max_batch_size):
        synthetic_dataset = model.sample(num_samples=n_samples)
        assert isinstance(synthetic_dataset, pd.DataFrame)
        assert synthetic_dataset.shape[0] == model.segment_length*n_samples
        assert synthetic_dataset.shape[1] == model.feat_dim + 2


def test_fit_method():
    model = create_sample_model()
    # Check to see if fit function raises errors for different batch size configs
    with pytest.raises(ValueError):
        model.fit(batch_size=-1)
    with pytest.raises(ValueError):
        model.fit(batch_size=10.01)
    with pytest.raises(ValueError):
        model.fit(batch_size=model.dataset.shape[0]+1)

    # Check to see if fit function raises errors for different lr configs
    with pytest.raises(ValueError):
        model.fit(batch_size=1, lr=-1)

    # Check to see if fit function raises errors for different epoch configs
    with pytest.raises(ValueError):
        model.fit(batch_size=1, epochs=-1)
    with pytest.raises(TypeError):
        model.fit(batch_size=1, epochs=10.10)

    # Check to see that shape of trackers matches that of shape of epochs
    max_epochs = 2
    for n_epochs in range(max_epochs):
        model = create_sample_model()
        model.fit(batch_size=16, epochs=n_epochs+1, verbose=False)
        assert isinstance(model.total_loss_tracker, np.float32)
        assert isinstance(model.kl_loss_tracker, np.float32)
        assert isinstance(model.reconstruction_loss_tracker, np.float32)
        assert len(model.kl_losses_all) == n_epochs+1
        assert len(model.rec_losses_all) == n_epochs+1
        assert len(model.total_losses_all) == n_epochs+1

