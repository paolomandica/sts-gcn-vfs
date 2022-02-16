model = dict(
    type='SiamSkeletonGCN',
    backbone=dict(
        type='STGCN',
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='ntu-rgb+d', strategy='spatial')),
    sim_head=dict(
        type='SimSiamHead',
        in_channels=256,
        norm_cfg=dict(type='SyncBN'),
        aggregation_time_in=38,
        aggregation_time_out=10,
        aggregation_joints_in=25,
        aggregation_joints_out=1,
        num_projection_fcs=3,
        projection_mid_channels=256,
        projection_out_channels=256,
        num_predictor_fcs=2,
        predictor_mid_channels=128,
        predictor_out_channels=256,
        with_norm=True,
        loss_feat=dict(type='CosineSimLoss', negative=False),
        spatial_type='avg'
    ),
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
    videos_per_gpu=64,
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
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer = dict(type='Adam', lr=1e-04, weight_decay=1e-05)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[50, 80, 120, 150])
total_epochs = 200
checkpoint_config = dict(interval=5)
evaluation = dict(interval=5, metrics=['top_k_accuracy'])
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 project='skeleton',
                 entity='sapienzavideocontrastive',
                 dir='wandb',
                 config=dict(
                     model=model,
                     train_pipeline=train_pipeline,
                     data=data,
                     optimizer=optimizer,
                     optimizer_config=optimizer_config,
                     lr_config=lr_config,
                     total_epochs=total_epochs
                 )
             ),
             interval=10)
    ])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/siam_stgcn_80e_ntu60_xsub_keypoint_3d/'
load_from = None
resume_from = None
find_unused_parameters = False
workflow = [('train', 1)]
