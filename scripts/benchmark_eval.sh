# 12 refinement iterations (like training)

#8 threads: ----------------------------------------
#      img:[320, 448], refinement_iters:12  |  194.8 
python scripts/benchmark_eval.py --model 'checkpoints/train_memsave/archive_models/noMemsave_bs12_models/chairs.pth' --image_size 320  448 --refinement_iters=12 --evaluation_iters=1000 --no_4d_corr --num_corr_channels=4 --no_4d_agg
#8 threads: ----------------------------------------
#      img:[320, 448], refinement_iters:12  |  208.9
python scripts/benchmark_eval.py --model 'checkpoints/train_memsave/archive_models/memsave_bs12_models/chairs.pth'   --image_size 320  448 --refinement_iters=12 --evaluation_iters=1000 --no_4d_corr --num_corr_channels=4 --no_4d_agg --alternate_corr
#8 threads: -----------------------------------------   
#      img:[512, 1024], refinement_iters:12  |  624.1   
python scripts/benchmark_eval.py --model 'checkpoints/train_memsave/archive_models/noMemsave_bs1_models/chairs.pth'  --image_size 512 1024 --refinement_iters=12 --evaluation_iters=1000 --no_4d_corr --num_corr_channels=4 --no_4d_agg
#8 threads: -----------------------------------------   
#      img:[512, 1024], refinement_iters:12  |  789.0   
python scripts/benchmark_eval.py --model 'checkpoints/train_memsave/archive_models/memsave_bs1_models/chairs.pth'    --image_size 512 1024 --refinement_iters=12 --evaluation_iters=1000 --no_4d_corr --num_corr_channels=4 --no_4d_agg --alternate_corr


# 32 refinement iterations (used for evlation of models)

#8 threads: ----------------------------------------
#      img:[320, 448], refinement_iters:32  |  315.4
python scripts/benchmark_eval.py --model 'checkpoints/train_memsave/archive_models/noMemsave_bs12_models/chairs.pth' --image_size 320  448 --refinement_iters=32 --evaluation_iters=1000 --no_4d_corr --num_corr_channels=4 --no_4d_agg
#8 threads: ----------------------------------------
#      img:[320, 448], refinement_iters:32  |  328.7
python scripts/benchmark_eval.py --model 'checkpoints/train_memsave/archive_models/memsave_bs12_models/chairs.pth'   --image_size 320  448 --refinement_iters=32 --evaluation_iters=1000 --no_4d_corr --num_corr_channels=4 --no_4d_agg --alternate_corr
#8 threads: -----------------------------------------   
#      img:[512, 1024], refinement_iters:32  |  775.4   
python scripts/benchmark_eval.py --model 'checkpoints/train_memsave/archive_models/noMemsave_bs1_models/chairs.pth'  --image_size 512 1024 --refinement_iters=32 --evaluation_iters=1000 --no_4d_corr --num_corr_channels=4 --no_4d_agg
#8 threads: -----------------------------------------   
#      img:[512, 1024], refinement_iters:32  |  947.5   
python scripts/benchmark_eval.py --model 'checkpoints/train_memsave/archive_models/memsave_bs1_models/chairs.pth'    --image_size 512 1024 --refinement_iters=32 --evaluation_iters=1000 --no_4d_corr --num_corr_channels=4 --no_4d_agg --alternate_corr

