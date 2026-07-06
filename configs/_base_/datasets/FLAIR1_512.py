# dataset settings
FLAIR1_type = 'FLAIR1_my_dataset'
FLAIR1_root = '/sdc/hwb/data/FLAIR1_512/'
FLAIR1_crop_size = (512, 512)


FLAIR1_val_pipeline = [  # 测试数据加载管道
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

val_FLAIR1 = dict(
        type=FLAIR1_type,
        data_root=FLAIR1_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=FLAIR1_val_pipeline)


