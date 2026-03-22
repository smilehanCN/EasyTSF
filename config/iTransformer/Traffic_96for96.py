exp_conf = dict(
    model_name="iTransformer",
    dataset_name='Weather',

    hist_len=96,
    pred_len=96,

    output_attention=False,
    d_model=512,
    d_ff=512,
    dropout=0.1,
    factor=3,
    n_heads=8,
    activation='gelu',
    e_layers=3,

    batch_size=16,
    max_epochs=30,
    lr=0.001,
    # lr_scheduler="WSD",
    # lr_warmup_end_epochs=5,
    # lr_stable_end_epochs=10,
    # es_patience=20,
)
