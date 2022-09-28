#$ Parameters: 8345734
#$ Validation (clean) EPE: 1.297001, 1px: 0.897865, 3px: 0.956408, 5px: 0.969679
#$ Validation (final) EPE: 2.600432, 1px: 0.849058, 3px: 0.919285, 5px: 0.939524
python evaluate.py --model './checkpoints/repo_model/sepflow_things.pth' --dataset 'sintel'

#$ Validation KITTI: 4.606222, 15.931453
python evaluate.py --model './checkpoints/repo_model/sepflow_things.pth' --dataset 'kitti'
