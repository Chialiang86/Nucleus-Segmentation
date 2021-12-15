# dataset settings
# dataset_type = 'CocoDataset'
# data_root = 'data/coco/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# modified (multi)
# data setting
dataset_type = 'CocoDataset'
data_root = 'dataset/'
classes = ('nucleus',)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomCrop', crop_size=(600, 600)),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(600, 600), # modified
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data_root = 'dataset/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        img_prefix=data_root,
        classes=classes,
        ann_file=data_root + 'train_annotation.json',
        type=dataset_type, 
        pipeline=train_pipeline, 
    ),
    val=dict(
        img_prefix=data_root,
        classes=classes,
        ann_file=data_root + 'val_annotation.json',
        type=dataset_type, 
        pipeline=test_pipeline, 
    ),
    test=dict(
        img_prefix=data_root + 'test/',
        classes=classes,
        ann_file=data_root + 'test_img_ids.json',
        type=dataset_type, 
        pipeline=test_pipeline, 
    )
)

evaluation = dict(metric=['bbox', 'segm']) # modified 'bbox', 
