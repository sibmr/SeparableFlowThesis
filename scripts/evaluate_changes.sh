python evaluate.py --model './checkpoints/original_model/things.pth' --dataset 'kitti'
#$ number of parameters: 8345734
#$ Validation KITTI: 8.836170, 27.598330
python evaluate.py --model './checkpoints/original_model/things.pth' --dataset 'sintel'
#$ Validation (clean) EPE: 1.708842, 1px: 0.831499, 3px: 0.936761, 5px: 0.957700
#$ Validation (final) EPE: 3.085266, 1px: 0.782289, 3px: 0.896234, 5px: 0.924516

python evaluate.py --model './checkpoints/train_no4dcorr/models/things.pth' --dataset 'kitti'   --no_4d_corr
#$ number of parameters: 7602246
#$ Validation KITTI: 11.306159, 34.673533
python evaluate.py --model './checkpoints/train_no4dcorr/models/things.pth' --dataset 'sintel'  --no_4d_corr
#$ Validation (clean) EPE: 1.934901, 1px: 0.802626, 3px: 0.926860, 5px: 0.951095
#$ Validation (final) EPE: 3.557808, 1px: 0.752827, 3px: 0.886614, 5px: 0.917748

python evaluate.py --model './checkpoints/train_K4/models/things.pth' --dataset 'kitti'         --no_4d_corr --num_corr_channels=4
#$ number of parameters: 7605922
#$ Validation KITTI: 11.842352, 38.292256
python evaluate.py --model './checkpoints/train_K4/models/things.pth' --dataset 'sintel'        --no_4d_corr --num_corr_channels=4
#$ Validation (clean) EPE: 1.998620, 1px: 0.796342, 3px: 0.923432, 5px: 0.948191
#$ Validation (final) EPE: 3.529886, 1px: 0.747387, 3px: 0.882230, 5px: 0.913739

python evaluate.py --model './checkpoints/train_no4dagg/models/things.pth' --dataset 'kitti'    --no_4d_corr --num_corr_channels=4 --no_4d_agg
#$ number of parameters: 7557454
#$ Validation KITTI: 10.421225, 32.323205
python evaluate.py --model './checkpoints/train_no4dagg/models/things.pth' --dataset 'sintel'   --no_4d_corr --num_corr_channels=4 --no_4d_agg
#$ Validation (clean) EPE: 1.849632, 1px: 0.812076, 3px: 0.929586, 5px: 0.952472
#$ Validation (final) EPE: 3.341073, 1px: 0.760815, 3px: 0.888469, 5px: 0.918409