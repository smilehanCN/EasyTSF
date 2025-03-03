exp_conf = dict(
    model_name="PatchTST",
    dataset_name='Weather',

    hist_len=336,
    pred_len=336,

    patch_len=16,
    stride=8,
    output_attention=False,
    d_model=256,
    dropout=0.2,
    factor=3,
    n_heads=8,
    activation='gelu',
    e_layers=3,

    batch_size=128,
    max_epochs=20,
    lr=0.0001,
    # lr_scheduler="WSD",
    # lr_warmup_end_epochs=5,
    # lr_stable_end_epochs=10,
    # es_patience=20,
)
