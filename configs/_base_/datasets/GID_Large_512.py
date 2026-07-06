# dataset settings
GID_Large_type = 'GID_Large_my_dataset'
GID_Large_root = '/sdc/hwb/data/GID_Large_RGB_512/'
GID_Large_crop_size = (512, 512)


GID_Large_test_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

val_GID_Large = dict(
        type=GID_Large_type,
        data_root=GID_Large_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=GID_Large_test_pipeline)

