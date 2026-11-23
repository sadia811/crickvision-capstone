auto_scale_lr = dict(base_batch_size=1024, enable=False)
backend_args = dict(backend='local')
base_lr = 0.004
codec = dict(
    input_size=(
        192,
        256,
    ),
    normalize=False,
    sigma=(
        4.9,
        5.66,
    ),
    simcc_split_ratio=2.0,
    type='SimCCLabel',
    use_dark=False)
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=190,
        switch_pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(type='RandomHalfBody'),
            dict(
                rotate_factor=15,
                scale_factor=[
                    0.85,
                    1.15,
                ],
                shift_factor=0.05,
                type='RandomBBoxTransform'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(
                encoder=dict(
                    input_size=(
                        192,
                        256,
                    ),
                    normalize=False,
                    sigma=(
                        4.9,
                        5.66,
                    ),
                    simcc_split_ratio=2.0,
                    type='SimCCLabel',
                    use_dark=False),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='mmdet.PipelineSwitchHook'),
]
data_mode = 'topdown'
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        interval=10,
        max_keep_ckpts=3,
        rule='greater',
        save_best='coco/AP',
        type='CheckpointHook'),
    logger=dict(interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = '/kaggle/working/crickvision_rtmpose_s_v3/rtmpose_s_pretrained_compatible_13kpt.pth'
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
max_epochs = 420
model = dict(
    backbone=dict(
        _scope_='mmdet',
        act_cfg=dict(type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.33,
        expand_ratio=0.5,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth',
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(type='BN'),
        out_indices=(4, ),
        type='CSPNeXt',
        widen_factor=0.5),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        final_layer_kernel_size=7,
        gau_cfg=dict(
            act_fn='SiLU',
            drop_path=0.0,
            dropout_rate=0.0,
            expansion_factor=2,
            hidden_dims=256,
            pos_enc=False,
            s=128,
            use_rel_bias=False),
        in_channels=512,
        in_featuremap_size=(
            6,
            8,
        ),
        input_size=(
            192,
            256,
        ),
        loss=dict(
            beta=10.0,
            label_softmax=True,
            type='KLDiscretLoss',
            use_target_weight=True),
        out_channels=13,
        simcc_split_ratio=2.0,
        type='RTMCCHead'),
    test_cfg=dict(flip_test=True),
    type='TopdownPoseEstimator')
optim_wrapper = dict(
    optimizer=dict(lr=0.0005, type='AdamW', weight_decay=0.0),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        T_max=105,
        begin=105,
        by_epoch=True,
        convert_to_iter_based=True,
        end=210,
        eta_min=2.5e-05,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=True, seed=42)
resume = False
stage2_num_epochs = 30
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file=
        '/kaggle/working/crickvision_rtmpose_s_v3/clean_annotations/test.json',
        data_mode='topdown',
        data_prefix=dict(
            img='/kaggle/working/crickvision_rtmpose_s_v3/roboflow_coco/test/'
        ),
        metainfo=dict(
            from_file=
            '/kaggle/working/crickvision_rtmpose_s_v3/config/crickvision_pose13.py'
        ),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/kaggle/working/crickvision_rtmpose_s_v3/clean_annotations/test.json',
    nms_mode='none',
    score_mode='bbox',
    type='CocoMetric')
train_cfg = dict(by_epoch=True, max_epochs=210, val_interval=5)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file=
        '/kaggle/working/crickvision_rtmpose_s_v3/clean_annotations/train.json',
        data_mode='topdown',
        data_prefix=dict(
            img='/kaggle/working/crickvision_rtmpose_s_v3/roboflow_coco/train/'
        ),
        metainfo=dict(
            from_file=
            '/kaggle/working/crickvision_rtmpose_s_v3/config/crickvision_pose13.py'
        ),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(type='RandomHalfBody'),
            dict(
                rotate_factor=30,
                scale_factor=[
                    0.75,
                    1.25,
                ],
                shift_factor=0.1,
                type='RandomBBoxTransform'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(
                transforms=[
                    dict(p=0.05, type='Blur'),
                    dict(p=0.05, type='MedianBlur'),
                    dict(
                        max_height=0.3,
                        max_holes=1,
                        max_width=0.3,
                        min_height=0.1,
                        min_holes=1,
                        min_width=0.1,
                        p=0.3,
                        type='CoarseDropout'),
                ],
                type='Albumentation'),
            dict(
                encoder=dict(
                    input_size=(
                        192,
                        256,
                    ),
                    normalize=False,
                    sigma=(
                        4.9,
                        5.66,
                    ),
                    simcc_split_ratio=2.0,
                    type='SimCCLabel',
                    use_dark=False),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=False,
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(type='RandomHalfBody'),
    dict(
        rotate_factor=80,
        scale_factor=[
            0.6,
            1.4,
        ],
        type='RandomBBoxTransform'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        transforms=[
            dict(p=0.1, type='Blur'),
            dict(p=0.1, type='MedianBlur'),
            dict(
                max_height=0.4,
                max_holes=1,
                max_width=0.4,
                min_height=0.2,
                min_holes=1,
                min_width=0.2,
                p=1.0,
                type='CoarseDropout'),
        ],
        type='Albumentation'),
    dict(
        encoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(type='RandomHalfBody'),
    dict(
        rotate_factor=60,
        scale_factor=[
            0.75,
            1.25,
        ],
        shift_factor=0.0,
        type='RandomBBoxTransform'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        transforms=[
            dict(p=0.1, type='Blur'),
            dict(p=0.1, type='MedianBlur'),
            dict(
                max_height=0.4,
                max_holes=1,
                max_width=0.4,
                min_height=0.2,
                min_holes=1,
                min_width=0.2,
                p=0.5,
                type='CoarseDropout'),
        ],
        type='Albumentation'),
    dict(
        encoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file=
        '/kaggle/working/crickvision_rtmpose_s_v3/clean_annotations/val.json',
        data_mode='topdown',
        data_prefix=dict(
            img='/kaggle/working/crickvision_rtmpose_s_v3/roboflow_coco/valid/'
        ),
        metainfo=dict(
            from_file=
            '/kaggle/working/crickvision_rtmpose_s_v3/config/crickvision_pose13.py'
        ),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/kaggle/working/crickvision_rtmpose_s_v3/clean_annotations/val.json',
    nms_mode='none',
    score_mode='bbox',
    type='CocoMetric')
val_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/kaggle/working/crickvision_rtmpose_s_v3/work_dir'
