# evaluation of original repo model with gma
# number of parameters: 8886407
python evaluate.py --model './checkpoints/train_originalGMA/models/things.pth' --dataset 'kitti' --use_gma
#$ Validation KITTI: 4.794522, 16.583632
python evaluate.py --model './checkpoints/train_originalGMA/models/things.pth' --dataset 'sintel' --use_gma
#$ Validation (clean) EPE: 1.272462, 1px: 0.901442, 3px: 0.957327, 5px: 0.969645
#$ Validation (final) EPE: 2.665907, 1px: 0.851505, 3px: 0.920519, 5px: 0.939741

# evaluation of paper-revised model with gma
# number of parameters: 8098127
python evaluate.py --model './checkpoints/train_no4daggGMA/models/things.pth' --dataset 'kitti'  --no_4d_corr --num_corr_channels=4 --no_4d_agg --use_gma
#$ Validation KITTI: 11.235443, 33.981523
python evaluate.py --model './checkpoints/train_no4daggGMA/models/things.pth' --dataset 'sintel' --no_4d_corr --num_corr_channels=4 --no_4d_agg --use_gma
#$ Validation (clean) EPE: 1.959904, 1px: 0.813692, 3px: 0.927790, 5px: 0.950695
#$ Validation (final) EPE: 3.639553, 1px: 0.759616, 3px: 0.884756, 5px: 0.914409