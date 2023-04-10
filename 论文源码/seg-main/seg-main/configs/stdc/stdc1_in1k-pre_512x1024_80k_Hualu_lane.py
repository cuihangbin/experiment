_base_ = [
    '../_base_/models/stdc_ZM.py', '../_base_/datasets/Hualu_lane.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
lr_config = dict(warmup='linear', warmup_iters=1000)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
)

# model = dict(
#     backbone=dict(
#         backbone_cfg=dict(
#             init_cfg=dict(
#                 type='Pretrained', checkpoint='./pretrained/stdc1.pth'))))
