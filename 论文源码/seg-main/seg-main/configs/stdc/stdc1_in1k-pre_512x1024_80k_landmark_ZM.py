_base_ = [
    '../_base_/models/stdc_ZM.py', '../_base_/datasets/landmark_ZM.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
lr_config = dict(warmup='linear', warmup_iters=1000)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
)

model = dict(
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(
                type='Pretrained', checkpoint='./pretrained/stdc1.pth'))))
