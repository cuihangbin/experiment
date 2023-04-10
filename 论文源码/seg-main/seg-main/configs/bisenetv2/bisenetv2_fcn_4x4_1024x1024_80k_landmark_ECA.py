_base_ = [
    '../_base_/models/bisenetv2_landmark_ECA.py',
    '../_base_/datasets/landmark_ZM.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.05)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
