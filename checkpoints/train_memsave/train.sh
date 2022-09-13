# Experiment with batch size 12 and (H,W) = (320,448) 
# Before eval: 21693MiB
# After eval: 22307MiB
python train.py --experiment_name='train_memsave' --stage='chairs'                                                            --run_name='chairs' --gpu='0' --num_steps 100000 --batchSize 12 --testBatchSize=4 --lr=0.0004   --image_size 320 448  --wdecay 0.0001   --freeze_bn=0 --gamma=0.8 --no_4d_corr --num_corr_channels=4 --no_4d_agg --alternate_corr_backward    --start_epoch=0  --thread=16 
# After eval: 23047MiB
#python train.py --experiment_name='train_memsave' --stage='chairs'                                                            --run_name='chairs' --gpu='0' --num_steps 100000 --batchSize 12 --testBatchSize=4 --lr=0.0004   --image_size 320 448  --wdecay 0.0001   --freeze_bn=0 --gamma=0.8 --no_4d_corr --num_corr_channels=4 --no_4d_agg                              --start_epoch=0  --thread=16 


# Experiment with batch size 1 and (H,W)=(512,1024)
# 16429MiB
# 5 min
# python train.py --experiment_name='train_memsave' --stage='chairs'                                                            --run_name='chairs' --gpu='0' --num_steps 100000 --batchSize 2 --testBatchSize=4 --lr=0.0004   --image_size 512 1024  --wdecay 0.0001   --freeze_bn=0 --gamma=0.8 --no_4d_corr --num_corr_channels=4 --no_4d_agg  --start_epoch=0  --thread=16 
# 15043MiB
# 25 min
# python train.py --experiment_name='train_memsave' --stage='chairs'                                                            --run_name='chairs' --gpu='0' --num_steps 100000 --batchSize 2 --testBatchSize=4 --lr=0.0004   --image_size 512 1024  --wdecay 0.0001   --freeze_bn=0 --gamma=0.8 --no_4d_corr --num_corr_channels=4 --no_4d_agg --alternate_corr_backward  --start_epoch=0  --thread=16 

# Before eval: 16425MiB
# After eval: 16425MiB
# 5 min
# python train.py --experiment_name='train_memsave' --stage='chairs'                                                            --run_name='chairs' --gpu='0' --num_steps 100000 --batchSize 1 --testBatchSize=4 --lr=0.0004   --image_size 512 1024  --wdecay 0.0001   --freeze_bn=0 --gamma=0.8 --no_4d_corr --num_corr_channels=4 --no_4d_agg  --start_epoch=0  --thread=16 
# Before eval: 8339MiB
# After eval: 10713MiB
# 25 min
# python train.py --experiment_name='train_memsave' --stage='chairs'                                                            --run_name='chairs' --gpu='0' --num_steps 100000 --batchSize 1 --testBatchSize=4 --lr=0.0004   --image_size 512 1024  --wdecay 0.0001   --freeze_bn=0 --gamma=0.8 --no_4d_corr --num_corr_channels=4 --no_4d_agg --alternate_corr_backward  --start_epoch=0  --thread=16 



#python train.py --experiment_name='train_memsave' --stage='things' --weights 'checkpoints/train_memsave/models/chairs.pth'    --run_name='things' --gpu='0' --num_steps 100000 --batchSize 8  --testBatchSize=4 --lr 0.000125 --image_size 448 768  --wdecay 0.0001   --freeze_bn=1 --gamma=0.8 --no_4d_corr --num_corr_channels=4 --no_4d_agg --alternate_corr_backward    

# Only C+T for now
# python train.py --experiment_name='train_originalLongerChairs' --stage='sintel' --weights 'checkpoints/train_originalLongerChairs/things.pth' --run_name='sintel' --gpu='0,1,2,3' --num_steps 100000 --batchSize 8  --testBatchSize=4 --lr 0.000125 --image_size 384 832  --wdecay 0.00001  --freeze_bn=1 --gamma=0.85    
# python train.py --experiment_name='train_originalLongerChairs' --stage='kitti'  --weights 'checkpoints/train_originalLongerChairs/sintel.pth' --run_name='kitti'  --gpu='0,1,2,3' --num_steps 50000  --batchSize 8  --testBatchSize=4 --lr 0.0001   --image_size 320 1024 --wdecay 0.00001  --freeze_bn=1 --gamma=0.85    