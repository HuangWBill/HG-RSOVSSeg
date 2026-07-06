# dataset settings
Globe230k_type = 'Globe230k_my_dataset'
Globe230k_root = '/sdc/hwb/data/Globe230k_512/'
Globe230k_crop_size = (512, 512)


Globe230k_test_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

test_Globe230k = dict(
        type=Globe230k_type,
        data_root=Globe230k_root,
        data_prefix=dict(img_path='img_dir/val_test', seg_map_path='ann_dir/val_test'),
        pipeline=Globe230k_test_pipeline)

