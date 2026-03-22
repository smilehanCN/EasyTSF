exp_conf = dict(
    model_name="iTransformer",
    dataset_name='ETTh1',

    hist_len=96,
    pred_len=96,

    output_attention=False,
    d_model=256,
    d_ff=256,
    dropout=0.1,
    factor=3,
    n_heads=8,
    activation='gelu',
    e_layers=2,

    batch_size=32,
    max_epochs=20,
    lr=0.0001,
)
