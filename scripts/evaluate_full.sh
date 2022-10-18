# Evaluate full training runs

# original + GMA + 100K chairs
python evaluate.py --output_path './test_flows/originalGMA' --model './checkpoints/train_originalGMA/models/sintel.pth'  --dataset 'sintel_test'  --use_gma
#$ Validation (clean) EPE: 0.653848, 1px: 0.922428, 3px: 0.969420, 5px: 0.980194
#$ Validation (final) EPE: 1.002394, 1px: 0.886479, 3px: 0.948263, 5px: 0.966085
python evaluate.py --model './checkpoints/train_originalGMA/models/sintel.pth' --dataset 'sintel' --use_gma
#$ Validation KITTI: 0.574540, 1.255321
python evaluate.py --model './checkpoints/train_originalGMA/models/kitti.pth'  --dataset 'kitti'  --use_gma

# original + 100K chairs
#$ python evaluate.py --model './checkpoints/train_originalLongerChairs/models/sintel.pth' --dataset 'sintel'
#$ Validation (clean) EPE: 0.686154, 1px: 0.918248, 3px: 0.967785, 5px: 0.979239
#$ Validation (final) EPE: 1.054699, 1px: 0.881393, 3px: 0.945974, 5px: 0.964269
#$ python evaluate.py --model './checkpoints/train_originalLongerChairs/models/kitti.pth'  --dataset 'kitti'
#$ Validation KITTI: 0.575549, 1.283686
python evaluate.py --output_path './test_flows/original' --model './checkpoints/train_originalLongerChairs/models/sintel.pth'  --dataset 'sintel_test'

# final ablation
#$ python evaluate.py --model './checkpoints/ablationsLongerChairs/train_no4dagg/models/sintel.pth' --dataset 'sintel'  --no_4d_corr --num_corr_channels=4 --no_4d_agg
#$ Validation (clean) EPE: 1.025676, 1px: 0.871131, 3px: 0.952066, 5px: 0.969212
#$ Validation (final) EPE: 1.474674, 1px: 0.832674, 3px: 0.925180, 5px: 0.949996
#$ python evaluate.py --model './checkpoints/ablationsLongerChairs/train_no4dagg/models/kitti.pth'  --dataset 'kitti'   --no_4d_corr --num_corr_channels=4 --no_4d_agg
#$ Validation KITTI: 0.732579, 1.832944
python evaluate.py --output_path './test_flows/no4dagg' --model './checkpoints/ablationsLongerChairs/train_no4dagg/models/sintel.pth' --dataset 'sintel_test'  --no_4d_corr --num_corr_channels=4 --no_4d_agg

