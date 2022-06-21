# python evaluate.py --model './checkpoints/train_no4dagg/models/things.pth' --dataset 'kitti'    --no_4d_corr --num_corr_channels=4 --no_4d_agg
#$ number of parameters: 7557454
#$ Validation KITTI: 10.421221, 32.323226
#$ 2279MiB

# python evaluate.py --model './checkpoints/train_no4dagg/models/things.pth' --dataset 'kitti'    --no_4d_corr --num_corr_channels=4 --no_4d_agg --alternate_corr
#$ number of parameters: 7557454
#$ Validation KITTI: 10.421211, 32.323232
#$ 1975MiB

python evaluate.py --model './checkpoints/train_no4dagg/models/things.pth' --dataset 'sintel'    --no_4d_corr --num_corr_channels=4 --no_4d_agg
#$ number of parameters: 7557454
#$ Validation (clean) EPE: 1.849631, 1px: 0.812076, 3px: 0.929586, 5px: 0.952472
#$ Validation (final) EPE: 3.341078, 1px: 0.760815, 3px: 0.888469, 5px: 0.918408
#$ 1041/1041 [14:57<00:00,  1.16it/s]
#$ 2165MiB

python evaluate.py --model './checkpoints/train_no4dagg/models/things.pth' --dataset 'sintel'    --no_4d_corr --num_corr_channels=4 --no_4d_agg --alternate_corr
#$ Validation (clean) EPE: 1.849630, 1px: 0.812076, 3px: 0.929586, 5px: 0.952472
#$ Validation (final) EPE: 3.341045, 1px: 0.760815, 3px: 0.888469, 5px: 0.918409
#$ 1041/1041 [29:20<00:00,  1.69s/it]
#$ 1743MiB