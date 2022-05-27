# python evaluate.py --model './checkpoints/original_model/kitti.pth' --dataset 'kitti'
# output$ Validation KITTI: 0.623273, 1.368164
python evaluate.py --model './checkpoints/original_model/kitti.pth' --dataset 'sintel'
# Validation (clean) EPE: 3.478742, 1px: 0.754892, 3px: 0.889513, 5px: 0.919135
# Validation (final) EPE: 4.811809, 1px: 0.712205, 3px: 0.852326, 5px: 0.887256
# python evaluate.py --model './checkpoints/original_model/things.pth' --dataset 'kitti'
# output$ Validation KITTI: 8.836172, 27.598426
# python evaluate.py --model './checkpoints/original_model/things.pth' --dataset 'sintel'
# output$ EPE: Validation (final) EPE: 3.085267, 1px: 0.782290, 3px: 0.896234, 5px: 0.924516
