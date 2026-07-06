
_base_ = [
    "./Potsdam_RGB_512.py",
    "./LoveDA_512.py",
    "./GID_Large_512.py",
    "./Globe230k_512.py",
    "./FLAIR1_512.py",
    "./OpenEarthMap_512.py",
    "./LandCover_ai_512.py",
]

# dataset settings
dataset_type = 'OpenEarthMap_my_dataset'
data_root = r'/sdc/hwb/data/OpenEarthMap_512/'
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type="MutiConcatDataset",
        datasets=[{{_base_.val_Potsdam}}, {{_base_.val_LoveDA}}, {{_base_.val_GID_Large}},
                  {{_base_.test_Globe230k}}, {{_base_.val_FLAIR1}}, {{_base_.val_OpenEarthMap}},
                  {{_base_.val_LandCover_ai}}]
        ))
test_dataloader = val_dataloader

val_evaluator = dict(
    type="DGIoUMetric",
    iou_metrics=["mIoU"],
    dataset_keys=["Potsdam", "LoveDA", "GID_Large", "Globe230k", "FLAIR1", "OpenEarthMap",
                  "LandCover_ai"],
)

test_evaluator = val_evaluator