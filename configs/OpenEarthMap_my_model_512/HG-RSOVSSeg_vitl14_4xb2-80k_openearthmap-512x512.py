_base_ = [
    '../_base_/models/hg_rsovsseg_vitb16.py',
    '../_base_/datasets/OpenEarthMap_my_dataset_512.py',
    '../_base_/default_runtime_iter.py', '../_base_/schedules/schedule_80k_iter.py',
]

custom_imports = dict(imports=['projects.HG-RSOVSSeg.mmseg.datasets',
                               'projects.HG-RSOVSSeg.mmseg.decode_heads',
                               'projects.HG-RSOVSSeg.mmseg.necks',
                               'projects.HG-RSOVSSeg.mmseg.segmentors',
                               'projects.HG-RSOVSSeg.mmseg.image_encoder',
                               'projects.HG-RSOVSSeg.mmseg.text_encoder',
                               'projects.HG-RSOVSSeg.mmseg.evaluation.dg_metrics'
                               ])
crop_size = (512,512)
data_preprocessor = dict(size=crop_size)
# model_cfg
model = dict(data_preprocessor=data_preprocessor,
             # pretrained='/sdd/hwb/code/mmsegmentation/pretrain/CLIP/ViT-L-14-new.pt',
             # pretrained='/sdd/hwb/code/mmsegmentation/pretrain/CLIP/ViT-L-14-336px-new.pt',
             # pretrained='/sdd/hwb/code/mmsegmentation/pretrain/RemoteCLIP/RemoteCLIP-ViT-L-14.pt',
             # pretrained='/sdd/hwb/code/mmsegmentation/pretrain/GeoRSCLIP/RS5M_ViT-L-14.pt',
             pretrained='/sdd/hwb/code/mmsegmentation/pretrain/GeoRSCLIP/RS5M_ViT-L-14-336.pt',
             # pretrained='/sdd/hwb/code/mmsegmentation/pretrain/SkyCLIP/SkyCLIP_ViT_L14-new.pt',
             pretrained_stage='NODecode',
             text_encoder=dict(
                 embed_dim=768,
                 context_length=77,
                 vocab_size=49408,
                 transformer_width=768,
                 transformer_heads=12,
                 transformer_layers=12,
                 frozen='frozen_all',  #'frozen_all'
             ),
             image_encoder=dict(
                 embed_dim=768,
                 out_idx=[7,15, 23],   # [5, 11, 17, 23][7,15, 23],
                 image_resolution=512,
                 vision_layers=24,
                 vision_heads=16,
                 vision_width=1024,
                 vision_patch_size=14,
                 frozen='frozen_all',  #'attention'
             ),
             neck=dict(
                 type='FA_Neck',
                 channels=1024,
                 out_channels=[128, 64],   # [256, 128, 64],
                 up_kernel_size=[2, 4],     # [2, 4, 8]
                 up_stride=[2, 4],     # [2, 4, 8]
             ),
             decode_head=dict(
                 type='HD_Head',
                 channels=768,
                 in_channels=768,
                 key_channels=512,
                 dropout=0.0,
                 decoder_dims=[128, 64],   # [256, 128, 64],
             ),
             train_cfg=dict(dataset_name='OpenEarthMap', vocabulary=None),
             test_cfg=dict(mode='whole', dataset_name=["Potsdam", "LoveDA", "GID_Large", "Globe230k", "FLAIR1",
                                                       "OpenEarthMap", "LandCover_ai"], vocabulary=None)
)

find_unused_parameters=True
randomness = dict(seed=0)


train_cfg = dict(val_interval=80000)

train_dataloader = dict(   # 训练数据加载器的配置
    batch_size=1,  # 每一个GPU的batch size大小
    num_workers=1  # 为每一个GPU预读取数据的进程个数
)