model = dict(
    type='SkeletonGCN',
    backbone=dict(
        type='STSGCN',
        input_channels=3,
        input_time_frame=30,
        st_gcnn_dropout=0.1,
        joints_to_consider=25,
        siamese=False,
        pretrained='./work_dirs/siam_stsgcn_300_ntu60_xsub_keypoint_3d/epoch_180.pth',
        freeze=True),
    cls_head=dict(
        type='STGCNHead',
        num_classes=60,
        in_channels=256,
        loss_cls=dict(type='CrossEntropyLoss')),
    train_cfg=None,
    test_cfg=None)

dataset_type = 'PoseDataset'
ann_file_train = '/data_volume/data/ntu60/annot_file/xsub/train.pkl'
ann_file_val = '/data_volume/data/ntu60/annot_file/xsub/val.pkl'
train_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=112,
    workers_per_gpu=8,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix='',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix='',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix='',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001,
                 nesterov=True)  # dict(type='Adam', lr=1e-02, weight_decay=1e-05)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[15, 25, 35, 50])
total_epochs = 50
checkpoint_config = dict(interval=5)
evaluation = dict(interval=5, metrics=['top_k_accuracy'])
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='WandbLoggerHook',
        #      init_kwargs=dict(
        #          project='skeleton',
        #          entity='sapienzavideocontrastive',
        #          dir='wandb',
        #          config=dict(
        #              model=model,
        #              train_pipeline=train_pipeline,
        #              data=data,
        #              optimizer=optimizer,
        #              optimizer_config=optimizer_config,
        #              lr_config=lr_config,
        #              total_epochs=total_epochs
        #          )
        #      ),
        #      interval=10)
    ])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/stsgcn_80e_ntu60_xsub_keypoint_3d/'
load_from = None
resume_from = None
workflow = [('train', 1)]
