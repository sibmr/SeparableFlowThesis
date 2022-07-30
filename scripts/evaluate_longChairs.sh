python evaluate.py --model './checkpoints/ablationsLongerChairs/train_no4dcorr/models/things.pth' --dataset 'kitti'     --no_4d_corr
#$ Validation KITTI: 11.124899, 35.389611
python evaluate.py --model './checkpoints/ablationsLongerChairs/train_no4dcorr/models/things.pth' --dataset 'sintel'    --no_4d_corr
#$ Validation (clean) EPE: 1.934373, 1px: 0.802038, 3px: 0.926010, 5px: 0.950131
#$ Validation (final) EPE: 3.514598, 1px: 0.754259, 3px: 0.885131, 5px: 0.915757


python evaluate.py --model './checkpoints/ablationsLongerChairs/train_K4/models/things.pth' --dataset 'kitti'   --no_4d_corr --num_corr_channels=4
#$ Validation KITTI: 10.289394, 30.966938
python evaluate.py --model './checkpoints/ablationsLongerChairs/train_K4/models/things.pth' --dataset 'sintel'  --no_4d_corr --num_corr_channels=4
#$ Validation (clean) EPE: 1.780403, 1px: 0.841380, 3px: 0.935228, 5px: 0.954906
#$ Validation (final) EPE: 3.245068, 1px: 0.793751, 3px: 0.894524, 5px: 0.920630

python evaluate.py --model './checkpoints/ablationsLongerChairs/train_no4dagg/models/things.pth' --dataset 'kitti'  --no_4d_corr --num_corr_channels=4 --no_4d_agg
#$ Validation KITTI: 10.502013, 31.964073
python evaluate.py --model './checkpoints/ablationsLongerChairs/train_no4dagg/models/things.pth' --dataset 'sintel' --no_4d_corr --num_corr_channels=4 --no_4d_agg
#$ Validation (clean) EPE: 1.812874, 1px: 0.831972, 3px: 0.933467, 5px: 0.954338
#$ Validation (final) EPE: 3.329740, 1px: 0.780799, 3px: 0.892769, 5px: 0.920434