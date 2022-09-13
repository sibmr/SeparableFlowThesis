# 12 refinement iterations (like training)
python benchmark_eval.py --model 'checkpoints/train_memsave/archive_models/noMemsave_bs12_models/chairs.pth' --image_size 320  448 --refinement_iters=12 --evaluation_iters=1000 --no_4d_corr --num_corr_channels=4 --no_4d_agg
python benchmark_eval.py --model 'checkpoints/train_memsave/archive_models/memsave_bs12_models/chairs.pth'   --image_size 320  448 --refinement_iters=12 --evaluation_iters=1000 --no_4d_corr --num_corr_channels=4 --no_4d_agg --alternate_corr
python benchmark_eval.py --model 'checkpoints/train_memsave/archive_models/noMemsave_bs1_models/chairs.pth'  --image_size 512 1024 --refinement_iters=12 --evaluation_iters=1000 --no_4d_corr --num_corr_channels=4 --no_4d_agg
python benchmark_eval.py --model 'checkpoints/train_memsave/archive_models/memsave_bs1_models/chairs.pth'    --image_size 512 1024 --refinement_iters=12 --evaluation_iters=1000 --no_4d_corr --num_corr_channels=4 --no_4d_agg --alternate_corr
# 32 refinement iterations (used for evlation of models)
python benchmark_eval.py --model 'checkpoints/train_memsave/archive_models/noMemsave_bs12_models/chairs.pth' --image_size 320  448 --refinement_iters=32 --evaluation_iters=1000 --no_4d_corr --num_corr_channels=4 --no_4d_agg
python benchmark_eval.py --model 'checkpoints/train_memsave/archive_models/memsave_bs12_models/chairs.pth'   --image_size 320  448 --refinement_iters=32 --evaluation_iters=1000 --no_4d_corr --num_corr_channels=4 --no_4d_agg --alternate_corr
python benchmark_eval.py --model 'checkpoints/train_memsave/archive_models/noMemsave_bs1_models/chairs.pth'  --image_size 512 1024 --refinement_iters=32 --evaluation_iters=1000 --no_4d_corr --num_corr_channels=4 --no_4d_agg
python benchmark_eval.py --model 'checkpoints/train_memsave/archive_models/memsave_bs1_models/chairs.pth'    --image_size 512 1024 --refinement_iters=32 --evaluation_iters=1000 --no_4d_corr --num_corr_channels=4 --no_4d_agg --alternate_corr

