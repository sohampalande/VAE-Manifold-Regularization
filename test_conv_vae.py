import numpy as np
import pandas as pd
from models.tsvae_conv import ConvTimeSeriesVAE


def create_sample_dataset(time_length=5000, num_features=5):
    # Create things that will fill in dataframe
    time_steps = np.array([np.linspace(0, 1, time_length)]).T
    features = np.random.normal(loc=0, scale=1, size=(time_length, num_features))
    # Create names of dataframe
    time_name = 'time'
    feature_names = [str(i) for i in range(num_features)]
    # Create pandas dataframe with the data
    dataset = pd.DataFrame(data=np.concatenate([time_steps, features], axis=1),
                           columns=[time_name]+feature_names)
    return dataset, time_name, feature_names


def test_constructor():
    # Create a dummy dataset to run tests on
    dataset, time_name, feature_names = create_sample_dataset()
    # Check for a variety of segment lengths that we get
    model = ConvTimeSeriesVAE(dataset=dataset, time_column=time_name, feature_names=feature_names,
                              segment_length=50)
    assert isinstance(model.dataset, np.ndarray)
    assert model.dataset.shape[0] == dataset.shape[0]-model.segment_length
    assert model.dataset.shape[1] == dataset.shape[1]-1
    assert model.dataset.shape[2] == model.segment_length


def test_create_encoder():
    pass


def test_create_decoder():
    pass


def test_encoder_forward_pass():
    pass


def test_decoder_forward_pass():
    pass


def test_vae_forward_pass():
    pass


def test_sampling_method():
    pass


def test_fit_method():
    pass

