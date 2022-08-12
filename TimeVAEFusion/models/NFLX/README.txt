PARAMETERS:

    seq_len = dataset.shape[-1]
    feat_dim = dataset.shape[1]    
    latent_dim = 8
    hidden_layer_sizes = [100, 200]
    epochs = 55
    seed = 1    # 1, 2, 3, 4, 5
    batch_size = 32
    lr=0.001
    reconstruction_wt = 4
    kernel_size = 3
    val_set = 0.1