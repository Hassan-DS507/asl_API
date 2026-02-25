class CFG:
    seed = 42
    max_len = 384
    dim = 192
    batch_size = 64 * 8
    epochs = 300
    lr = 5e-4 * 8
    weight_decay = 0.1
    warmup = 0
    awp = True
    awp_lambda = 0.2
    awp_start_epoch = 15
    dropout_start_epoch = 15