large_scale_io_conf = dict(
    use_mmap=True,
    cache_npz_as_npy=True,
    precompute_window_index=True,
)


cost_conf = dict(
    dataset_name='cost',
    var_num=7,
    freq=60,
    data_split=[2000, 2000, 2000],
)

Pseudo_conf = dict(
    dataset_name='Pseudo',
    var_num=1,
    freq=6*24,
    data_split=[6000, 2000, 2000],
)

WDIR2022_conf = dict(
    dataset_name='WDIR2022',
    var_num=4232,
    freq=10,
    data_split=[31536, 10512, 10512],
    **large_scale_io_conf,
)

WSPD2022_conf = dict(
    dataset_name='WSPD2022',
    var_num=4232,
    freq=10,
    data_split=[31536, 10512, 10512],
    **large_scale_io_conf,
)

SZWeatherDemo_conf = dict(
    dataset_name='SZWeatherDemo',
    var_num=4,
    freq=10,
    data_split=[28979, 9660, 9660],
)

ETTh1_conf = dict(
    dataset_name='ETTh1',
    var_num=7,
    freq=60,
    data_split=[8640, 2880, 2880],
)

ETTh2_conf = dict(
    dataset_name='ETTh2',
    var_num=7,
    freq=60,
    data_split=[8640, 2880, 2880],
)

ETTm1_conf = dict(
    dataset_name='ETTm1',
    var_num=7,
    freq=15,
    data_split=[34560, 11520, 11520],
)

ETTm2_conf = dict(
    dataset_name='ETTm2',
    var_num=7,
    freq=15,
    data_split=[34560, 11520, 11520],
)

ECL_conf = dict(
    dataset_name='ECL',
    var_num=321,
    freq=60,
    data_split=[18412, 2632, 5260],
    **large_scale_io_conf,
)

Weather_conf = dict(
    dataset_name='Weather',
    var_num=21,
    freq=10,
    data_split=[36887, 5270, 10539],
)

Traffic_conf = dict(
    dataset_name='Traffic',
    var_num=862,
    freq=60,
    data_split=[12280, 1756, 3508],
    **large_scale_io_conf,
)

Illness_conf = dict(
    dataset_name='Illness',
    var_num=7,
    freq=60 * 24 * 7,
    data_split=[676, 97, 193],
)

SolarEnergy_conf = dict(
    dataset_name='SolarEnergy',
    var_num=137,
    freq=10,
    data_split=[36792, 5256, 10512],
    **large_scale_io_conf,
)

PEMS03_conf = dict(
    dataset_name='PEMS03',
    var_num=358,
    freq=5,
    data_split=[15724, 5242, 5242],
    **large_scale_io_conf,
)

PEMS04_conf = dict(
    dataset_name='PEMS04',
    var_num=307,
    freq=5,
    data_split=[10196, 3398, 3398],
    **large_scale_io_conf,
)

PEMS07_conf = dict(
    dataset_name='PEMS07',
    var_num=883,
    freq=5,
    data_split=[16934, 5645, 5645],
    **large_scale_io_conf,
)

PEMS08_conf = dict(
    dataset_name='PEMS08',
    var_num=170,
    freq=5,
    data_split=[10714, 3571, 3571],
    **large_scale_io_conf,
)

wind_conf = dict(
    dataset_name='wind',
    var_num=7,
    freq=15,
    data_split=[34071, 4867, 9735],
)

BJAQ_conf = dict(
    dataset_name='BJAQ',
    var_num=7,
    freq=60,
    data_split=[25200, 3600, 7200],
)
