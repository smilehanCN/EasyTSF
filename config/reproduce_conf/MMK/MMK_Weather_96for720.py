exp_conf = dict(
    model_name="MMK",
    dataset_name='Weather',

    hist_len=96,
    pred_len=96,
    hidden_dim=64,

    layer_type="MoK",
    layer_hp=[['TaylorKAN', 4], ['TaylorKAN', 4], ['JacobiKAN', 4], ['JacobiKAN', 4]],
    layer_num=3,

    lr=0.0001,
    max_epochs=20,
    es_patience=20,
    batch_size=32,

    lr_scheduler="WSD",
    lr_warmup_end_epochs=5,
    lr_stable_end_epochs=10,
)
