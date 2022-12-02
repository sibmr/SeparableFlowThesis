"""
    original Separable Flow evaluate.sh file, evaluate.sh files of this thesis can be found in scripts/
"""

python evaluate.py --model './checkpoints/sepflow_sintel.pth' --dataset 'sintel'
python evaluate.py --model './checkpoints/sepflow_kitti.pth' --dataset 'kitti'
