norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[122.7709383, 116.7460125, 104.09373615],
    std=[68.5005327, 66.6321579, 70.3231630],
    pad_val=0,
    seg_pad_val=255)
# model_cfg
model = dict(
    type='TexT_EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='/home/hwb/.cache/clip/ViT-B-16.pt',
    image_input=False,
    pretrained_stage='NODecode',
    text_encoder=dict(
        type='CLIP_Text',
        embed_dim=512,  # 768
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        templates='vild',
        cache_feature=False,
    ),
    image_encoder=dict(
        type='CLIP_Image',
        embed_dim=512,
        out_origin=False,
        out_idx=[2, 5, 8, 11],
        use_proj=True,
        image_resolution=512,
        vision_layers=12,
        vision_heads=12,
        vision_width=768,
        vision_patch_size=16,
    ),
    neck=dict(
        type='FA_Neck',
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        type='HD_Head',
        channels=512,
        in_channels=512,
        num_classes=5,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(dataset_name='Potsdam', vocabulary=None),
    test_cfg=dict(mode='whole', dataset_name=["Potsdam","LoveDA", "GID_Fine", "Globe230k"], vocabulary=None))