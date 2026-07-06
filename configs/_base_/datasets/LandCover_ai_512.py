# dataset settings
LandCover_ai_type = 'LandCover_ai_my_dataset'
LandCover_ai_root = '/sdc/hwb/data/LandCover_ai_512/'
LandCover_ai_crop_size = (512, 512)


LandCover_ai_test_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

val_LandCover_ai = dict(
        type=LandCover_ai_type,
        data_root=LandCover_ai_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=LandCover_ai_test_pipeline)

