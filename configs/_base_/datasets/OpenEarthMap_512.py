# dataset settings
OpenEarthMap_type = 'OpenEarthMap_my_dataset'
OpenEarthMap_root = '/sdc/hwb/data/OpenEarthMap_512/'
OpenEarthMap_crop_size = (512, 512)


OpenEarthMap_test_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

val_OpenEarthMap = dict(
        type=OpenEarthMap_type,
        data_root=OpenEarthMap_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=OpenEarthMap_test_pipeline)

