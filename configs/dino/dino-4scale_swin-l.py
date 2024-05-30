_base_ = './dino-4scale_r50_8xb2-12e_coco.py'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
num_levels = 4
num_classes = 91
cat_ids = list(range(91))
model = dict(
    num_feature_levels=num_levels,
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True,
        sampler=False,
        convert_weights=True,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained),
        ),
    neck=dict(in_channels=[384, 768, 1536], num_outs=num_levels, bias=True),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels))),
    bbox_head=dict(num_classes=num_classes),
    dn_cfg=dict(num_classes=num_classes+1))
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="PackDetInputs"),
]
train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False),
    dataset=dict(cat_ids=cat_ids,pipeline=train_pipeline)
)
val_dataloader = dict(dataset=dict(cat_ids=cat_ids,pipeline=train_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(cat_ids=cat_ids)
test_evaluator = val_evaluator