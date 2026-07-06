# dataset settings
LoveDA_type = 'LoveDA_my_dataset'
LoveDA_root = '/sdc/hwb/data/LoveDA_512/'
LoveDA_crop_size = (512, 512)


LoveDA_test_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

val_LoveDA = dict(
        type=LoveDA_type,
        data_root=LoveDA_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=LoveDA_test_pipeline)

