PARAMETERS:

    seq_len = dataset.shape[-1]
    feat_dim = dataset.shape[1]    
    latent_dim = 8
    hidden_layer_sizes = [100, 250]
    reconstruction_wt = 3
    kernel_size = 3
    seed = 2    # 1, 2, 3, 4, 5
    set_seed(seed)
    val_set = 0.1
    epochs = 55
    batch_size = 32
    lr=0.001